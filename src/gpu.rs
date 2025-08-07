use std::ops::Deref;
use std::{error, fmt, io};

use ash::prelude::VkResult;
use ash::{khr, vk};

use crate::graphics;

pub struct Gpu {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    device_name: String,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    display_dimensions: DisplayDimensions,
    swapchain_loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    renderpass: vk::RenderPass,
    pipeline: Pipeline,
    control: Control,
    images: Vec<ImageBuffer>,
    storage_image: Option<StorageImage>,
}

#[derive(Clone, Copy)]
struct DisplayDimensions {
    width: u32,
    height: u32,
}

struct Buffer<T> {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    mapped_memory: *mut T,
    host_coherent: bool,
    descriptor_set: vk::DescriptorSet,
}

struct Control {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    present_complete_semaphore: vk::Semaphore,
    rendering_complete_semaphore: vk::Semaphore,
}

struct Pipeline {
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    viewports: [vk::Viewport; 1],
    scissors: [vk::Rect2D; 1],
    layout: vk::PipelineLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
}

const IMAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_UNORM;
const MIN_IMAGES: u32 = 2;

struct ImageBuffer {
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    image: vk::Image,
    command_buffer: vk::CommandBuffer,
    param_buffer: Buffer<ShaderParams>,
}

struct StorageImage {
    image: vk::Image,
    buffer_memory: vk::DeviceMemory,
    view: vk::ImageView,
}

#[repr(C)]
struct ShaderParams {
    timecode: f32,
    width: f32,
    height: f32,
    max_distance: f32,
}

pub struct RenderFeedback {
    pub swapchain_suboptimal: bool,
    pub queue_suboptimal: bool,
}

struct ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    val: Option<T>,
    rollback: Option<F>,
}

pub struct Configuration<'a> {
    pub shader: Option<&'a [u8]>,
    pub store_image: bool,
}

impl Gpu {
    pub fn init(conf: Configuration) -> Result<Gpu, GpuError> {
        unsafe {
            // Vulkan init code is long an messy - here's an attempt to make it somewhat atomic and simplify cleanup on error.
            // Everything depends on everything, and instead of having a struct with lots of Option fields,
            // init everything in one place and return a fully initialized GPU instance.
            // If anything goes wrong, ScopeRollback will call desctuctors.
            let entry = ash::Entry::load()?;
            let instance = Self::init_vk(&entry)?;
            let instance = ScopeRollback::new(instance, |instance| instance.destroy_instance(None));
            let display_loader = khr::display::Instance::new(&entry, &instance);
            let surface_loader = khr::surface::Instance::new(&entry, &instance);

            let (physical_device, device_name, graphics_queue) = Self::find_device(&instance)?;
            let (surface, display_dimensions, display_name) =
                Self::create_display(&display_loader, physical_device)?;
            let surface = ScopeRollback::new(surface, |surface| {
                surface_loader.destroy_surface(surface, None)
            });
            let device = Self::create_device(&instance, physical_device, graphics_queue)?;
            let device = ScopeRollback::new(device, |device| device.destroy_device(None));

            let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
            let swapchain = Self::create_swapchain(
                &surface_loader,
                &swapchain_loader,
                physical_device,
                graphics_queue,
                *surface.deref(),
                display_dimensions,
            )?;
            let swapchain = ScopeRollback::new(swapchain, |swapchain| {
                swapchain_loader.destroy_swapchain(swapchain, None)
            });
            let swapchain_images = swapchain_loader.get_swapchain_images(*swapchain.deref())?;

            let renderpass = Self::create_renderpass(&device)?;
            let renderpass = ScopeRollback::new(renderpass, |renderpass| {
                let device: &ash::Device = &device;
                device.destroy_render_pass(renderpass, None)
            });
            let pipeline = Self::create_pipeline_layout(
                &device,
                display_dimensions,
                *renderpass.deref(),
                swapchain_images.len() as u32,
                conf.shader,
            )?;
            let control = Self::create_control(&device, graphics_queue)?;
            let control = ScopeRollback::new(control, |control| control.destroy(&device));

            let memory_properties = {
                let instance: &ash::Instance = &instance;
                instance.get_physical_device_memory_properties(physical_device)
            };
            let storage_image = if conf.store_image {
                let storage_image =
                    Self::create_storage_image(&device, &memory_properties, display_dimensions)?;
                let storage_image = ScopeRollback::new(storage_image, |storage_image| {
                    storage_image.destroy(&device)
                });
                Some(storage_image)
            } else {
                None
            };
            let images = Self::create_image_buffers(
                &device,
                swapchain_images,
                &memory_properties,
                *renderpass.deref(),
                {
                    let control: &Control = &control;
                    control.command_pool
                },
                display_dimensions,
                &pipeline,
            )?;
            if let Some(storage_image) = &storage_image {
                storage_image.add_to_descriptor_sets(&device, &images);
            }
            let device_name = format!(
                "GPU: {}, display: {} ({}x{})",
                device_name, display_name, display_dimensions.width, display_dimensions.height
            );

            // All is good - consume instances and defuse the scope rollback.
            let surface = surface.consume();
            let swapchain = swapchain.consume();
            let control = control.consume();
            let renderpass = renderpass.consume();
            let storage_image = storage_image.map(|storage_image| storage_image.consume());
            Ok(Gpu {
                _entry: entry,
                instance: instance.consume(),
                device: device.consume(),
                device_name,
                surface_loader,
                surface,
                display_dimensions,
                swapchain_loader,
                swapchain,
                renderpass,
                pipeline,
                control,
                images,
                storage_image,
            })
        }
    }

    pub fn device_name(&self) -> &str {
        self.device_name.as_str()
    }

    fn init_vk(entry: &ash::Entry) -> VkResult<ash::Instance> {
        let app_name = c"Hypno Toadface";
        let engine_name = c"hypno-toadface";
        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let enabled_extensions = [khr::display::NAME.as_ptr(), khr::surface::NAME.as_ptr()];
        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(create_flags)
            .enabled_extension_names(&enabled_extensions);
        unsafe { entry.create_instance(&create_info, None) }
    }

    fn find_device(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, String, u32), GpuError> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = unsafe { instance.get_physical_device_properties(device) };
                if props.limits.max_uniform_buffer_range
                    < std::mem::size_of::<ShaderParams>() as u32
                {
                    return None;
                }
                let queue_index = Self::find_graphics_queue(instance, device)?;

                let device_name = props
                    .device_name_as_c_str()
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
                // TODO: allow to specify a device name filter/regex?
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                };
                Some((device, device_name, queue_index, score))
            })
            .max_by(|(_, _, _, a), (_, _, _, b)| a.cmp(b));
        let (device, name, queue_index) = if let Some((device, name, queue_index, _score)) = device
        {
            (device, name, queue_index)
        } else {
            return Err("Device not found".into());
        };
        Ok((device, name, queue_index))
    }

    fn find_graphics_queue(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<u32> {
        unsafe {
            instance
                .get_physical_device_queue_family_properties(device)
                .iter()
                .enumerate()
                .flat_map(|(index, queue)| {
                    if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) && queue.queue_count > 0
                    {
                        Some(index as u32)
                    } else {
                        None
                    }
                })
                .next()
        }
    }

    fn create_display(
        display_loader: &khr::display::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(vk::SurfaceKHR, DisplayDimensions, String), GpuError> {
        let displays =
            unsafe { display_loader.get_physical_device_display_properties(physical_device)? };
        let display = displays.first().ok_or("No displays found")?;

        let display_modes = unsafe {
            display_loader.get_display_mode_properties(physical_device, display.display)?
        };
        let display_mode = display_modes.first().ok_or("No display modes found")?;

        let display_planes = unsafe {
            display_loader.get_physical_device_display_plane_properties(physical_device)?
        };
        let display_plane_index = display_planes
            .iter()
            .enumerate()
            .find_map(|(i, _plane)| {
                let supported_displays = unsafe {
                    display_loader
                        .get_display_plane_supported_displays(physical_device, i as u32)
                        .ok()?
                };
                if supported_displays.contains(&display.display) {
                    return None;
                }
                // Use the first supported display.

                // TODO: check capabilities?
                /*
                let caps = display_loader
                    .get_display_plane_capabilities(
                        physical_device,
                        display_mode.display_mode,
                        i as u32,
                    )
                    .ok()?;
                */

                Some(i)
            })
            .ok_or("No compatible display modes found")?;

        let display_mode_create_info =
            vk::DisplayModeCreateInfoKHR::default().parameters(display_mode.parameters);
        let surface_display_mode = unsafe {
            display_loader.create_display_mode(
                physical_device,
                display.display,
                &display_mode_create_info,
                None,
            )?
        };

        let display_surface_create_info = vk::DisplaySurfaceCreateInfoKHR::default()
            .display_mode(surface_display_mode)
            .plane_index(display_plane_index as u32)
            .image_extent(display_mode.parameters.visible_region);
        let display_plane_surface = unsafe {
            display_loader.create_display_plane_surface(&display_surface_create_info, None)?
        };

        let display_name = unsafe {
            display
                .display_name_as_c_str()
                .unwrap_or(c"[unknown]")
                .to_string_lossy()
                .to_string()
        };

        let display_dimensions = DisplayDimensions {
            width: display_mode.parameters.visible_region.width,
            height: display_mode.parameters.visible_region.height,
        };
        Ok((display_plane_surface, display_dimensions, display_name))
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
    ) -> Result<ash::Device, vk::Result> {
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_index)
            .queue_priorities(&[1.0f32]);
        let enabled_extensions = [khr::swapchain::NAME.as_ptr()];
        let features = vk::PhysicalDeviceFeatures::default().shader_clip_distance(true);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&enabled_extensions)
            .enabled_features(&features);
        unsafe { instance.create_device(physical_device, &device_create_info, None) }
    }

    fn create_swapchain(
        surface_loader: &khr::surface::Instance,
        swapchain_loader: &khr::swapchain::Device,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
        surface: vk::SurfaceKHR,
        dimensions: DisplayDimensions,
    ) -> Result<vk::SwapchainKHR, GpuError> {
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
        };
        if !surface_capabilities
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            return Err("Opaque composite alpha not supported".into());
        }

        let device_supported = unsafe {
            surface_loader.get_physical_device_surface_support(
                physical_device,
                graphics_queue_index,
                surface,
            )?
        };
        if !device_supported {
            return Err("Device doesn't support surface".into());
        }

        // TODO: pick a presentation mode from a prioritized list?
        let supports_mailbox = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(physical_device, surface)?
                .contains(&vk::PresentModeKHR::MAILBOX)
        };
        let present_mode = if supports_mailbox {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        };

        // Use at least double buffering.
        let min_image_count = if surface_capabilities.max_image_count == 0 {
            surface_capabilities.min_image_count.max(MIN_IMAGES)
        } else {
            MIN_IMAGES.clamp(
                surface_capabilities.min_image_count,
                surface_capabilities.max_image_count,
            )
        };

        // Choose the best image format.
        let surface_formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)?
        };
        surface_formats
            .iter()
            .find(|format| {
                format.format == IMAGE_FORMAT
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .ok_or("Cannot find a supported surface format")?;

        let queue_family_indices = [graphics_queue_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(min_image_count)
            .image_format(IMAGE_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(dimensions.extent2d())
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .present_mode(present_mode);

        unsafe { Ok(swapchain_loader.create_swapchain(&swapchain_create_info, None)?) }
    }

    fn create_renderpass(device: &ash::Device) -> Result<vk::RenderPass, vk::Result> {
        let color_attachment_description = vk::AttachmentDescription::default()
            .format(IMAGE_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachments = [vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let subpasses = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments)];
        let attachments = [color_attachment_description];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        unsafe { device.create_render_pass(&render_pass_create_info, None) }
    }

    fn create_pipeline_layout(
        device: &ash::Device,
        dimensions: DisplayDimensions,
        renderpass: vk::RenderPass,
        images_count: u32,
        fragment_code: Option<&[u8]>,
    ) -> Result<Pipeline, GpuError> {
        let vertex_code: &[u8] = include_bytes!("../shaders/hypno-toadface.vert.spv");
        let fragment_code: &[u8] = match fragment_code {
            Some(fragment_code) => fragment_code,
            None => include_bytes!("../shaders/hypno-toadface.frag.spv"),
        };
        let vertex_code = ash::util::read_spv(&mut std::io::Cursor::new(vertex_code))?;
        let fragment_code = ash::util::read_spv(&mut std::io::Cursor::new(fragment_code))?;
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(vertex_code.as_slice());
        let fragment_shader_info =
            vk::ShaderModuleCreateInfo::default().code(fragment_code.as_slice());
        let vertex_shader_module = unsafe {
            device
                .create_shader_module(&vertex_shader_info, None)
                .map_err(|err| ("Vertex shader module error", err))?
        };
        let vertex_shader_module = ScopeRollback::new(vertex_shader_module, |module| unsafe {
            device.destroy_shader_module(module, None)
        });
        let fragment_shader_module = unsafe {
            device
                .create_shader_module(&fragment_shader_info, None)
                .map_err(|err| ("Fragment shader module error", err))?
        };
        let fragment_shader_module = ScopeRollback::new(fragment_shader_module, |module| unsafe {
            device.destroy_shader_module(module, None)
        });

        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(*vertex_shader_module.deref())
                .name(c"main")
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(*fragment_shader_module.deref())
                .name(c"main")
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        // Do not bind any vertex buffers to pipeline, the vertex buffer will generate them directly.
        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_attribute_descriptions(&[])
            .vertex_binding_descriptions(&[]);
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let scissors = [dimensions.render_area()];
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: dimensions.width as f32,
            height: dimensions.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .scissors(&scissors)
            .viewports(&viewports);

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::default()
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK);
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false);

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_state);

        let descriptor_pool_size = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(images_count),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(images_count),
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(images_count)
            .pool_sizes(&descriptor_pool_size);

        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };
        let descriptor_pool = ScopeRollback::new(descriptor_pool, |descriptor_pool| unsafe {
            device.destroy_descriptor_pool(descriptor_pool, None)
        });

        let params_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let texture_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let layout_bindings = [params_layout_binding, texture_layout_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&layout_info, None)? };
        let descriptor_set_layout =
            ScopeRollback::new(descriptor_set_layout, |descriptor_set_layout| unsafe {
                device.destroy_descriptor_set_layout(descriptor_set_layout, None)
            });

        let descriptor_set_layouts = [*descriptor_set_layout.deref()];
        let layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
        let layout = unsafe { device.create_pipeline_layout(&layout_create_info, None)? };
        let layout = ScopeRollback::new(layout, |layout| unsafe {
            device.destroy_pipeline_layout(layout, None)
        });

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(*layout.deref())
            .render_pass(renderpass);

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphic_pipeline_info],
                    None,
                )
                .map_err(|(_pipelines, err)| err)?
        };
        let pipeline = graphics_pipelines[0];

        Ok(Pipeline {
            vertex_shader_module: vertex_shader_module.consume(),
            fragment_shader_module: fragment_shader_module.consume(),
            pipeline,
            viewports,
            scissors,
            layout: layout.consume(),
            descriptor_pool: descriptor_pool.consume(),
            descriptor_set_layout: descriptor_set_layout.consume(),
        })
    }

    fn create_control(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Control, vk::Result> {
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        let command_pool = ScopeRollback::new(command_pool, |command_pool| unsafe {
            device.destroy_command_pool(command_pool, None)
        });
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = unsafe { device.create_fence(&fence_create_info, None)? };
        let fence = ScopeRollback::new(fence, |fence| unsafe { device.destroy_fence(fence, None) });
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let present_complete_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None)? };
        let present_complete_semaphore =
            ScopeRollback::new(present_complete_semaphore, |semaphore| unsafe {
                device.destroy_semaphore(semaphore, None)
            });
        let rendering_complete_semaphore =
            unsafe { device.create_semaphore(&semaphore_create_info, None)? };
        Ok(Control {
            queue,
            command_pool: command_pool.consume(),
            fence: fence.consume(),
            present_complete_semaphore: present_complete_semaphore.consume(),
            rendering_complete_semaphore,
        })
    }

    fn create_image_buffers(
        device: &ash::Device,
        swapchain_images: Vec<vk::Image>,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        dimensions: DisplayDimensions,
        pipeline: &Pipeline,
    ) -> Result<Vec<ImageBuffer>, GpuError> {
        let mut images = vec![];
        for image in swapchain_images {
            let image = Self::create_image_buffer(
                device,
                memory_properties,
                image,
                renderpass,
                command_pool,
                dimensions,
                pipeline,
            )
            .inspect_err(|_| {
                images.iter().for_each(|image: &ImageBuffer| {
                    image.destroy(device, command_pool, pipeline.descriptor_pool)
                });
            })?;
            images.push(image);
        }
        Ok(images)
    }

    fn create_image_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        image: vk::Image,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        dimensions: DisplayDimensions,
        pipeline: &Pipeline,
    ) -> Result<ImageBuffer, GpuError> {
        let component_mapping = vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::R)
            .g(vk::ComponentSwizzle::G)
            .b(vk::ComponentSwizzle::B)
            .a(vk::ComponentSwizzle::A);
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(IMAGE_FORMAT)
            .components(component_mapping)
            .subresource_range(subresource_range);
        let view = unsafe { device.create_image_view(&image_view_info, None)? };
        let view = ScopeRollback::new(view, |view| unsafe {
            device.destroy_image_view(view, None)
        });

        let attachments = [*view.deref()];
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(renderpass)
            .attachments(&attachments)
            .width(dimensions.width)
            .height(dimensions.height)
            .layers(1);
        let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None)? };
        let framebuffer = ScopeRollback::new(framebuffer, |framebuffer| unsafe {
            device.destroy_framebuffer(framebuffer, None)
        });

        let command_buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffers = unsafe { device.allocate_command_buffers(&command_buffers_info)? };
        let command_buffer = ScopeRollback::new(command_buffers[0], |_| unsafe {
            device.free_command_buffers(command_pool, &command_buffers)
        });

        let param_buffer: Buffer<ShaderParams> =
            Self::create_buffer::<ShaderParams>(device, memory_properties, pipeline)?;
        Ok(ImageBuffer {
            view: view.consume(),
            framebuffer: framebuffer.consume(),
            image,
            command_buffer: command_buffer.consume(),
            param_buffer,
        })
    }

    fn create_storage_image(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        dimensions: DisplayDimensions,
    ) -> Result<StorageImage, GpuError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(IMAGE_FORMAT)
            .extent(dimensions.extent3d())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&image_info, None)? };
        let image = ScopeRollback::new(image, |image| unsafe { device.destroy_image(image, None) });

        let memory_requirements = unsafe { device.get_image_memory_requirements(*image.deref()) };
        let (buffer_memory, _) = Self::allocate_memory(
            device,
            memory_properties,
            &memory_requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let buffer_memory = ScopeRollback::new(buffer_memory, |memory| unsafe {
            device.free_memory(memory, None)
        });
        unsafe { device.bind_image_memory(*image.deref(), *buffer_memory.deref(), 0)? };

        let component_mapping = vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::R)
            .g(vk::ComponentSwizzle::G)
            .b(vk::ComponentSwizzle::B)
            .a(vk::ComponentSwizzle::A);
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(*image.deref())
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(IMAGE_FORMAT)
            .components(component_mapping)
            .subresource_range(subresource_range);
        let view = unsafe { device.create_image_view(&image_view_info, None)? };

        Ok(StorageImage {
            image: image.consume(),
            buffer_memory: buffer_memory.consume(),
            view,
        })
    }

    fn allocate_memory(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        memory_requirements: &vk::MemoryRequirements,
        required_memory_properties: vk::MemoryPropertyFlags,
    ) -> Result<(vk::DeviceMemory, bool), GpuError> {
        // Most vendors provide a sorted list - with less features going first.
        // As soon as the right flag is found, this search will stop, so it should pick a memory
        // type with the closest match.
        let buffer_memory = memory_properties
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .flat_map(|(memory_type_index, memory_type)| {
                if memory_properties.memory_heaps[memory_type.heap_index as usize].size
                    < memory_requirements.size
                {
                    return None;
                }
                if ((1 << memory_type_index) & memory_requirements.memory_type_bits) == 0 {
                    return None;
                }
                let property_flags = memory_type.property_flags;
                if !property_flags.contains(required_memory_properties) {
                    return None;
                }
                let host_coherent = property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                let allocate_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index as u32);
                // Some buffers may fill up, in this case allocating memory can fail.
                let mem = unsafe { device.allocate_memory(&allocate_info, None).ok()? };

                Some((mem, host_coherent))
            })
            .next();
        if let Some((buffer_memory, host_coherent)) = buffer_memory {
            Ok((buffer_memory, host_coherent))
        } else {
            Err("Cannot find suitable memory".into())
        }
    }

    fn create_buffer<T>(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        pipeline: &Pipeline,
    ) -> Result<Buffer<T>, GpuError> {
        let size = std::mem::size_of::<T>() as u64;
        let required_memory_properties =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };
        let buffer = ScopeRollback::new(buffer, |buffer| unsafe {
            device.destroy_buffer(buffer, None)
        });
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(*buffer.deref()) };

        let (buffer_memory, host_coherent) = Self::allocate_memory(
            device,
            memory_properties,
            &memory_requirements,
            required_memory_properties,
        )?;
        let buffer_memory = ScopeRollback::new(buffer_memory, |memory| unsafe {
            device.free_memory(memory, None)
        });
        let mapped_memory = unsafe {
            device.bind_buffer_memory(*buffer.deref(), *buffer_memory.deref(), 0)?;
            device.map_memory(*buffer_memory.deref(), 0, size, vk::MemoryMapFlags::empty())?
        };
        let mapped_memory = mapped_memory as *mut T;
        let mapped_memory = ScopeRollback::new(mapped_memory, |_| unsafe {
            device.unmap_memory(*buffer_memory.deref())
        });

        let layouts = [pipeline.descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pipeline.descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info)? };
        let descriptor_set = descriptor_sets[0];
        let descriptor_set = ScopeRollback::new(descriptor_set, |descriptor_set| unsafe {
            let _ = device.free_descriptor_sets(pipeline.descriptor_pool, &[descriptor_set]);
        });

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(*buffer.deref())
            .offset(0)
            .range(size);
        let buffer_infos = [buffer_info];
        let write_buffer_info = vk::WriteDescriptorSet::default()
            .dst_set(*descriptor_set.deref())
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos);
        unsafe {
            device.update_descriptor_sets(&[write_buffer_info], &[]);
        }

        let mapped_memory = mapped_memory.consume();
        let buffer_memory = buffer_memory.consume();
        Ok(Buffer {
            buffer: buffer.consume(),
            buffer_memory,
            mapped_memory,
            descriptor_set: descriptor_set.consume(),
            host_coherent,
        })
    }

    fn copy_buffer_data(
        &self,
        image: &ImageBuffer,
        scene: &graphics::Scene,
    ) -> Result<(), vk::Result> {
        let timecode = scene.timecode as f32;
        let width = self.display_dimensions.width as f32 / 2.0;
        let height = self.display_dimensions.height as f32 / 2.0;
        let max_distance = width * width + height * height;
        let params = ShaderParams {
            timecode,
            width,
            height,
            max_distance,
        };
        image.param_buffer.write(&self.device, params)
    }

    fn save_last_frame(&self, image: &ImageBuffer) {
        let save_destination = if let Some(storage_image) = &self.storage_image {
            storage_image.image
        } else {
            return;
        };
        let copy_subresource = vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1);
        let copy_region = vk::ImageCopy::default()
            .src_subresource(copy_subresource)
            .dst_subresource(copy_subresource)
            .extent(self.display_dimensions.extent3d());
        unsafe {
            self.device.cmd_copy_image(
                image.command_buffer,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                save_destination,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );
        }
    }

    fn render_image(&self, image: &ImageBuffer, scene: &graphics::Scene) -> Result<(), GpuError> {
        let command_buffer = image.command_buffer;

        unsafe {
            self.device
                .wait_for_fences(&[self.control.fence], true, u64::MAX)?;
            self.device.reset_fences(&[self.control.fence])?;
            self.device.reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )?;
        }

        self.copy_buffer_data(image, scene)?;

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(command_buffer, &info)? };

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.renderpass)
            .framebuffer(image.framebuffer)
            .render_area(self.display_dimensions.render_area())
            .clear_values(&clear_values);

        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin,
                vk::SubpassContents::INLINE,
            );

            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );

            self.device
                .cmd_set_viewport(command_buffer, 0, &self.pipeline.viewports);
            self.device
                .cmd_set_scissor(command_buffer, 0, &self.pipeline.scissors);
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.layout,
                0,
                &[image.param_buffer.descriptor_set],
                &[0],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);

            self.save_last_frame(image);

            self.device.cmd_end_render_pass(command_buffer);

            self.device.end_command_buffer(command_buffer)?;
        }

        let wait_semaphores = [self.control.present_complete_semaphore];
        let signal_semaphores = [self.control.rendering_complete_semaphore];
        let command_buffers = [command_buffer];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let queue_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&command_buffers)
            .wait_dst_stage_mask(&wait_dst_stage_mask);

        unsafe {
            self.device.queue_submit(
                self.control.queue,
                &[queue_submit_info],
                self.control.fence,
            )?
        };
        Ok(())
    }

    pub fn render(&self, scene: &graphics::Scene) -> Result<RenderFeedback, GpuError> {
        unsafe {
            let (image_index, swapchain_suboptimal) = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.control.present_complete_semaphore,
                vk::Fence::null(),
            )?;

            if image_index as usize >= self.images.len() {
                return Err("Swapchain requested image {}, outside of range".into());
            }
            let image = &self.images[image_index as usize];
            self.render_image(image, scene)?;

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let wait_semaphores = [self.control.rendering_complete_semaphore];
            let queue_present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let queue_suboptimal = self
                .swapchain_loader
                .queue_present(self.control.queue, &queue_present_info)?;
            Ok(RenderFeedback {
                swapchain_suboptimal,
                queue_suboptimal,
            })
        }
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.control.queue);
            let _ = self.device.device_wait_idle();
            self.images.iter().for_each(|image| {
                image.destroy(
                    &self.device,
                    self.control.command_pool,
                    self.pipeline.descriptor_pool,
                )
            });
            if let Some(storage_image) = &self.storage_image {
                storage_image.destroy(&self.device);
            }
            self.control.destroy(&self.device);
            self.pipeline.destroy(&self.device);
            self.device.destroy_render_pass(self.renderpass, None);
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Control {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.rendering_complete_semaphore, None);
            device.destroy_semaphore(self.present_complete_semaphore, None);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl<T> Buffer<T> {
    fn write(&self, device: &ash::Device, val: T) -> Result<(), vk::Result> {
        unsafe { self.mapped_memory.write(val) };

        if self.host_coherent {
            return Ok(());
        };
        let flush_memory_ranges = vk::MappedMemoryRange::default()
            .memory(self.buffer_memory)
            .offset(0)
            .size(std::mem::size_of::<T>() as u64);
        unsafe { device.flush_mapped_memory_ranges(&[flush_memory_ranges])? };
        Ok(())
    }

    fn destroy(&self, device: &ash::Device, descriptor_pool: vk::DescriptorPool) {
        unsafe {
            let _ = device.free_descriptor_sets(descriptor_pool, &[self.descriptor_set]);
            device.unmap_memory(self.buffer_memory);
            device.free_memory(self.buffer_memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

impl ImageBuffer {
    fn destroy(
        &self,
        device: &ash::Device,
        command_pool: vk::CommandPool,
        descriptor_pool: vk::DescriptorPool,
    ) {
        unsafe {
            self.param_buffer.destroy(device, descriptor_pool);
            device.free_command_buffers(command_pool, &[self.command_buffer]);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.view, None);
        }
    }
}

impl StorageImage {
    fn add_to_descriptor_sets(&self, device: &ash::Device, images: &[ImageBuffer]) {
        images.iter().for_each(|image_buffer| {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(self.view)
                .image_layout(vk::ImageLayout::GENERAL);
            let image_infos = [image_info];
            let write_image_info = vk::WriteDescriptorSet::default()
                .dst_set(image_buffer.param_buffer.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&image_infos);
            unsafe { device.update_descriptor_sets(&[write_image_info], &[]) };
        });
    }

    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image(self.image, None);
            device.free_memory(self.buffer_memory, None);
            device.destroy_image_view(self.view, None);
        }
    }
}

impl Pipeline {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_shader_module(self.fragment_shader_module, None);
            device.destroy_shader_module(self.vertex_shader_module, None);
        }
    }
}

impl DisplayDimensions {
    fn extent2d(&self) -> vk::Extent2D {
        vk::Extent2D::default()
            .width(self.width)
            .height(self.height)
    }

    fn extent3d(&self) -> vk::Extent3D {
        vk::Extent3D::default()
            .width(self.width)
            .height(self.height)
            .depth(1)
    }

    fn render_area(&self) -> vk::Rect2D {
        let offset = vk::Offset2D::default().x(0).y(0);
        vk::Rect2D::default().offset(offset).extent(self.extent2d())
    }
}

impl<T, F> ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    fn new(val: T, rollback: F) -> ScopeRollback<T, F> {
        ScopeRollback {
            val: Some(val),
            rollback: Some(rollback),
        }
    }

    fn consume(mut self) -> T {
        self.rollback = None;
        self.val.take().unwrap()
    }
}

impl<T, F> Deref for ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val.as_ref().unwrap()
    }
}

impl<T, F> Drop for ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    fn drop(&mut self) {
        if let Some(val) = self.val.take()
            && let Some(rb) = self.rollback.take()
        {
            rb(val)
        }
    }
}

#[derive(Debug)]
pub enum GpuError {
    Internal(String),
    Loading(ash::LoadingError),
    Vk(String, vk::Result),
    Io(io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(ref msg) => f.write_str(msg),
            Self::Loading(ref e) => {
                write!(f, "Failed to init GPU: {e}")
            }
            Self::Vk(ref msg, ref e) => {
                if !msg.is_empty() {
                    write!(f, "Vulkan error: {msg} ({e})")
                } else {
                    write!(f, "Vulkan error: {e}")
                }
            }
            Self::Io(ref e) => {
                write!(f, "IO error: {e}")
            }
        }
    }
}

impl error::Error for GpuError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(ref _msg) => None,
            Self::Loading(ref e) => Some(e),
            Self::Vk(_, ref e) => Some(e),
            Self::Io(ref e) => Some(e),
        }
    }
}

impl From<ash::LoadingError> for GpuError {
    fn from(err: ash::LoadingError) -> GpuError {
        Self::Loading(err)
    }
}

impl From<vk::Result> for GpuError {
    fn from(err: vk::Result) -> GpuError {
        Self::Vk("".into(), err)
    }
}

impl From<(&str, vk::Result)> for GpuError {
    fn from(e: (&str, vk::Result)) -> GpuError {
        Self::Vk(e.0.to_string(), e.1)
    }
}

impl From<io::Error> for GpuError {
    fn from(err: io::Error) -> GpuError {
        Self::Io(err)
    }
}

impl From<&str> for GpuError {
    fn from(msg: &str) -> GpuError {
        Self::Internal(msg.to_string())
    }
}
