use std::{error, fmt, io};

use ash::prelude::VkResult;
use ash::{khr, vk};

use crate::graphics;

pub struct Gpu {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    device_name: String,
    display: Display,
    swapchain: vk::SwapchainKHR,
    renderpass: vk::RenderPass,
    pipeline: Pipeline,
    control: Control,
    images: Vec<ImageBuffer>,
    swapchain_extension: khr::swapchain::Device,
}

struct Display {
    width: u32,
    height: u32,
    surface: vk::SurfaceKHR,
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
const MIN_IMAGES: u32 = 16;

struct ImageBuffer {
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    command_buffer: vk::CommandBuffer,
    param_buffer: Buffer<ShaderParams>,
}

struct ShaderParams {
    _timecode: f32,
    _width: f32,
    _height: f32,
    _max_distance: f32,
}

pub struct RenderFeedback {
    pub swapchain_suboptimal: bool,
    pub queue_suboptimal: bool,
}

impl Gpu {
    pub fn init() -> Result<Gpu, GpuError> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = unsafe { Gpu::init_vk(&entry)? };
        let display_extension = khr::display::Instance::new(&entry, &instance);
        let surface_extension = khr::surface::Instance::new(&entry, &instance);

        let (physical_device, device_name, graphics_queue) =
            unsafe { Gpu::find_device(&instance)? };
        let (display, display_name) =
            unsafe { Gpu::create_display(&display_extension, physical_device)? };
        let device = unsafe { Gpu::create_device(&instance, physical_device, graphics_queue)? };
        let swapchain_extension = khr::swapchain::Device::new(&instance, &device);
        let swapchain = unsafe {
            Gpu::create_swapchain(
                &surface_extension,
                &swapchain_extension,
                physical_device,
                graphics_queue,
                &display,
            )?
        };
        let renderpass = unsafe { Gpu::create_renderpass(&device)? };
        // TODO: cleanup on error
        let pipeline = unsafe { Gpu::create_pipeline_layout(&device, &display, renderpass)? };
        let control = unsafe { Gpu::create_control(&device, graphics_queue)? };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let images = unsafe {
            Gpu::create_image_buffers(
                &swapchain_extension,
                &device,
                &memory_properties,
                swapchain,
                renderpass,
                control.command_pool,
                &display,
                &pipeline,
            )?
        };
        let device_name = format!(
            "GPU: {}, display: {} ({}x{})",
            device_name, display_name, display.width, display.height
        );
        Ok(Gpu {
            entry,
            instance,
            device,
            device_name,
            display,
            swapchain,
            renderpass,
            pipeline,
            control,
            images,
            swapchain_extension,
        })
    }

    pub fn device_name(&self) -> &str {
        self.device_name.as_str()
    }

    unsafe fn init_vk(entry: &ash::Entry) -> VkResult<ash::Instance> {
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
        entry.create_instance(&create_info, None)
    }

    unsafe fn find_device(
        instance: &ash::Instance,
    ) -> Result<(vk::PhysicalDevice, String, u32), GpuError> {
        let devices = instance.enumerate_physical_devices()?;
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = instance.get_physical_device_properties(device);
                // TODO: check UBO props.limits.max_uniform_buffer_range
                if false
                //props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                // || props.limits.max_per_stage_descriptor_storage_buffers < MAX_BINDINGS
                // TODO: add pipeline/descriptor set limitations
                {
                    return None;
                }
                let queue_index = Gpu::find_graphics_queue(instance, device)?;

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
            .max_by(|(_, _, _, a), (_, _, _, b)| a.cmp(&b));
        let (device, name, queue_index) = if let Some((device, name, queue_index, _score)) = device
        {
            (device, name, queue_index)
        } else {
            return Err("Device not found".into());
        };
        Ok((device, name, queue_index))
    }

    unsafe fn find_graphics_queue(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> Option<u32> {
        instance
            .get_physical_device_queue_family_properties(device)
            .iter()
            .enumerate()
            .flat_map(|(index, queue)| {
                if queue.queue_flags.contains(vk::QueueFlags::GRAPHICS) && queue.queue_count > 0 {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .next()
    }

    unsafe fn create_display(
        display_extension: &khr::display::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> Result<(Display, String), GpuError> {
        let displays = display_extension.get_physical_device_display_properties(physical_device)?;
        let display = displays.first().ok_or("No displays found")?;

        let display_modes =
            display_extension.get_display_mode_properties(physical_device, display.display)?;
        let display_mode = display_modes.first().ok_or("No display modes found")?;

        let display_planes =
            display_extension.get_physical_device_display_plane_properties(physical_device)?;
        let display_plane_index = display_planes
            .iter()
            .enumerate()
            .find_map(|(i, _plane)| {
                let supported_displays = display_extension
                    .get_display_plane_supported_displays(physical_device, i as u32)
                    .ok()?;
                if supported_displays.contains(&display.display) {
                    return None;
                }
                // Use the first supported display.

                // TODO: check capabilities?
                /*
                let caps = display_extension
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
        let surface_display_mode = display_extension.create_display_mode(
            physical_device,
            display.display,
            &display_mode_create_info,
            None,
        )?;

        let display_surface_create_info = vk::DisplaySurfaceCreateInfoKHR::default()
            .display_mode(surface_display_mode)
            .plane_index(display_plane_index as u32)
            .image_extent(display_mode.parameters.visible_region);
        let display_plane_surface =
            display_extension.create_display_plane_surface(&display_surface_create_info, None)?;

        let display_name = display
            .display_name_as_c_str()
            .unwrap_or(c"[unknown]")
            .to_string_lossy()
            .to_string();

        let display = Display {
            width: display_mode.parameters.visible_region.width,
            height: display_mode.parameters.visible_region.height,
            surface: display_plane_surface,
        };
        Ok((display, display_name))
    }

    unsafe fn create_device(
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
        instance.create_device(physical_device, &device_create_info, None)
    }

    unsafe fn create_swapchain(
        surface_extension: &khr::surface::Instance,
        swapchain_extension: &khr::swapchain::Device,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
        display: &Display,
    ) -> Result<vk::SwapchainKHR, GpuError> {
        let surface_capabilities = surface_extension
            .get_physical_device_surface_capabilities(physical_device, display.surface)?;
        if !surface_capabilities
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            return Err("opaque composite alpha not supported".into());
        }

        let device_supported = surface_extension.get_physical_device_surface_support(
            physical_device,
            graphics_queue_index,
            display.surface,
        )?;
        if !device_supported {
            return Err("Device doesn't support surface".into());
        }

        // TODO: pick a presentation mode from a prioritized list?
        let supports_mailbox = surface_extension
            .get_physical_device_surface_present_modes(physical_device, display.surface)?
            .contains(&vk::PresentModeKHR::MAILBOX);
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
        let surface_formats = surface_extension
            .get_physical_device_surface_formats(physical_device, display.surface)?;
        surface_formats
            .iter()
            .find(|format| {
                format.format == IMAGE_FORMAT
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .ok_or("Cannot find a supported surface format")?;

        let queue_family_indices = [graphics_queue_index];
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(display.surface)
            .min_image_count(min_image_count)
            .image_format(IMAGE_FORMAT)
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_extent(vk::Extent2D {
                width: display.width,
                height: display.height,
            })
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .present_mode(present_mode);

        Ok(swapchain_extension.create_swapchain(&swapchain_create_info, None)?)
    }

    unsafe fn create_image_buffers(
        swapchain_extension: &khr::swapchain::Device,
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        swapchain: vk::SwapchainKHR,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        display: &Display,
        pipeline: &Pipeline,
    ) -> Result<Vec<ImageBuffer>, GpuError> {
        let swapchain_images = swapchain_extension.get_swapchain_images(swapchain)?;
        swapchain_images
            .iter()
            .map(|image| {
                Gpu::create_image_buffer(
                    &device,
                    &memory_properties,
                    *image,
                    renderpass,
                    command_pool,
                    display,
                    pipeline,
                )
            })
            .collect::<Result<Vec<_>, _>>()
    }

    unsafe fn create_image_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        image: vk::Image,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        display: &Display,
        pipeline: &Pipeline,
    ) -> Result<ImageBuffer, GpuError> {
        // TODO: cleanup on error
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
        let view = device.create_image_view(&image_view_info, None)?;

        let attachments = [view];
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(renderpass)
            .attachments(&attachments)
            .width(display.width)
            .height(display.height)
            .layers(1);
        let framebuffer = device.create_framebuffer(&framebuffer_create_info, None)?;

        let command_buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer = device.allocate_command_buffers(&command_buffers_info)?[0];

        let param_buffer: Buffer<ShaderParams> =
            unsafe { Gpu::create_buffer::<ShaderParams>(&device, &memory_properties, &pipeline)? };
        Ok(ImageBuffer {
            view,
            framebuffer,
            command_buffer,
            param_buffer,
        })
    }

    unsafe fn create_renderpass(device: &ash::Device) -> Result<vk::RenderPass, vk::Result> {
        // TODO: cleanup on error
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
        device.create_render_pass(&render_pass_create_info, None)
    }

    unsafe fn create_buffer<T>(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        pipeline: &Pipeline,
    ) -> Result<Buffer<T>, GpuError> {
        let size = std::mem::size_of::<T>() as u64;
        let required_memory_properties =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = device.create_buffer(&buffer_create_info, None)?;
        let memory_requirements = device.get_buffer_memory_requirements(buffer);
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
                let mem = device.allocate_memory(&allocate_info, None).ok()?;

                Some((mem, host_coherent))
            })
            .next();

        let (buffer_memory, host_coherent) = if let Some(mem) = buffer_memory {
            mem
        } else {
            device.destroy_buffer(buffer, None);
            return Err("Cannot find suitable memory".into());
        };
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        let mapped_memory = device
            .map_memory(buffer_memory, 0, size as u64, vk::MemoryMapFlags::empty())
            .map_err(|err| {
                device.destroy_buffer(buffer, None);
                device.free_memory(buffer_memory, None);
                err
            })?;
        let mapped_memory = mapped_memory as *mut T;
        // TODO: unmap memory on cleanup.

        let layouts = [pipeline.descriptor_set_layout];
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pipeline.descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_allocate_info)?;
        let descriptor_set = descriptor_sets[0];

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(size as u64);
        let buffer_infos = [buffer_info];
        let write_descriptor = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&buffer_infos);
        device.update_descriptor_sets(&[write_descriptor], &[]);

        Ok(Buffer {
            buffer,
            buffer_memory,
            mapped_memory,
            descriptor_set,
            host_coherent,
        })
    }

    unsafe fn create_pipeline_layout(
        device: &ash::Device,
        display: &Display,
        renderpass: vk::RenderPass,
    ) -> Result<Pipeline, GpuError> {
        // TODO: cleanup on error
        let vertex_code: &[u8] = include_bytes!("../shaders/hypno-toadface.vert.spv");
        let fragment_code: &[u8] = include_bytes!("../shaders/hypno-toadface.frag.spv");
        let vertex_code = ash::util::read_spv(&mut std::io::Cursor::new(vertex_code))?;
        let fragment_code = ash::util::read_spv(&mut std::io::Cursor::new(fragment_code))?;
        let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(vertex_code.as_slice());
        let fragment_shader_info =
            vk::ShaderModuleCreateInfo::default().code(fragment_code.as_slice());
        let vertex_shader_module = device
            .create_shader_module(&vertex_shader_info, None)
            .map_err(|err| ("Vertex shader module error", err))?;
        let fragment_shader_module = device
            .create_shader_module(&fragment_shader_info, None)
            .map_err(|err| ("Fragment shader module error", err))?;

        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader_module)
                .name(c"main")
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader_module)
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

        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: display.width,
                height: display.height,
            },
        }];

        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: display.width as f32,
            height: display.height as f32,
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

        let descriptor_pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(MIN_IMAGES)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MIN_IMAGES)
            .pool_sizes(&descriptor_pool_size);

        // TODO: cleanup on error
        let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        let layout_bindings = [layout_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

        let descriptor_set_layouts = [descriptor_set_layout];
        let layout_create_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts);
        let layout = device.create_pipeline_layout(&layout_create_info, None)?;

        let graphic_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stage_create_infos)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_state_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .render_pass(renderpass);

        let graphics_pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[graphic_pipeline_info], None)
            .map_err(|(_pipelines, err)| err)?;
        let pipeline = graphics_pipelines[0];

        Ok(Pipeline {
            vertex_shader_module,
            fragment_shader_module,
            pipeline,
            viewports,
            scissors,
            layout,
            descriptor_pool,
            descriptor_set_layout,
        })
    }

    unsafe fn create_control(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Control, vk::Result> {
        let queue = device.get_device_queue(queue_family_index, 0);
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device.create_command_pool(&command_pool_info, None)?;
        let cleanup_err = |err| {
            device.destroy_command_pool(command_pool, None);
            err
        };
        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = device
            .create_fence(&fence_create_info, None)
            .map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_command_pool(command_pool, None);
            device.destroy_fence(fence, None);
            err
        };
        // TODO: cleanup unused resources
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let present_complete_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .map_err(cleanup_err)?;
        let rendering_complete_semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .map_err(cleanup_err)?;
        Ok(Control {
            queue,
            command_pool,
            fence,
            present_complete_semaphore,
            rendering_complete_semaphore,
        })
    }

    unsafe fn copy_buffer_data(
        &self,
        image_index: usize,
        scene: &graphics::Scene,
    ) -> Result<(), vk::Result> {
        // TODO: panic if index is out of range?
        // TODO: probably a good idea to wrap this in the Buffer.
        let buffer = &self.images[image_index].param_buffer;
        let timecode = scene.timecode / 100.0;
        let timecode = timecode - timecode.floor();
        let width = self.display.width as f32 / 2.0;
        let height = self.display.height as f32 / 2.0;
        let max_distance = width * width + height * height;
        let params = ShaderParams {
            _timecode: timecode as f32,
            _width: width,
            _height: height,
            _max_distance: max_distance,
        };
        buffer.mapped_memory.write(params);

        if buffer.host_coherent {
            return Ok(());
        };
        let flush_memory_ranges = vk::MappedMemoryRange::default()
            .memory(buffer.buffer_memory)
            .offset(0)
            .size(std::mem::size_of::<ShaderParams>() as u64);
        self.device
            .flush_mapped_memory_ranges(&[flush_memory_ranges])
    }

    unsafe fn render_image(
        &self,
        scene: &graphics::Scene,
        image_index: usize,
    ) -> Result<(), vk::Result> {
        let image = &self.images[image_index as usize];
        let command_buffer = image.command_buffer;

        self.device
            .wait_for_fences(&[self.control.fence], true, u64::MAX)?;
        self.device.reset_fences(&[self.control.fence])?;
        self.device.reset_command_buffer(
            command_buffer,
            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
        )?;

        self.copy_buffer_data(image_index as usize, scene)?;

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(command_buffer, &info)?;

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let render_area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: self.display.width,
                height: self.display.height,
            },
        };
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.renderpass)
            .framebuffer(image.framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);

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

        self.device.cmd_end_render_pass(command_buffer);
        self.device.end_command_buffer(command_buffer)?;

        let wait_semaphores = [self.control.present_complete_semaphore];
        let signal_semaphores = [self.control.rendering_complete_semaphore];
        let command_buffers = [command_buffer];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let queue_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&command_buffers)
            .wait_dst_stage_mask(&wait_dst_stage_mask);

        self.device
            .queue_submit(self.control.queue, &[queue_submit_info], self.control.fence)
    }

    pub fn render(&self, scene: &graphics::Scene) -> Result<RenderFeedback, GpuError> {
        unsafe {
            let (image_index, swapchain_suboptimal) = self.swapchain_extension.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.control.present_complete_semaphore,
                vk::Fence::null(),
            )?;

            self.render_image(scene, image_index as usize)?;

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let wait_semaphores = [self.control.rendering_complete_semaphore];
            let queue_present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&wait_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let queue_suboptimal = self
                .swapchain_extension
                .queue_present(self.control.queue, &queue_present_info)?;
            // TODO: move this into the cleanup function
            // self.device.queue_wait_idle(self.control.queue)?;
            Ok(RenderFeedback {
                swapchain_suboptimal,
                queue_suboptimal,
            })
        }
    }
}

#[derive(Debug)]
pub enum GpuError {
    InternalError(String),
    LoadingError(ash::LoadingError),
    VkError(String, vk::Result),
    IoError(io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GpuError::InternalError(ref msg) => f.write_str(msg),
            GpuError::LoadingError(ref e) => {
                write!(f, "Failed to init GPU: {}", e)
            }
            GpuError::VkError(ref msg, ref e) => {
                if !msg.is_empty() {
                    write!(f, "Vulkan error: {} ({})", msg, e)
                } else {
                    write!(f, "Vulkan error: {}", e)
                }
            }
            GpuError::IoError(ref e) => {
                write!(f, "IO error: {}", e)
            }
        }
    }
}

impl error::Error for GpuError {
    fn cause(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            GpuError::InternalError(ref _msg) => None,
            GpuError::LoadingError(ref e) => Some(e),
            GpuError::VkError(_, ref e) => Some(e),
            GpuError::IoError(ref e) => Some(e),
        }
    }
}

impl From<ash::LoadingError> for GpuError {
    fn from(err: ash::LoadingError) -> GpuError {
        GpuError::LoadingError(err)
    }
}

impl From<vk::Result> for GpuError {
    fn from(err: vk::Result) -> GpuError {
        GpuError::VkError("".into(), err)
    }
}

impl From<(&str, vk::Result)> for GpuError {
    fn from(e: (&str, vk::Result)) -> GpuError {
        GpuError::VkError(e.0.to_string(), e.1)
    }
}

impl From<io::Error> for GpuError {
    fn from(err: io::Error) -> GpuError {
        GpuError::IoError(err)
    }
}

impl From<&str> for GpuError {
    fn from(msg: &str) -> GpuError {
        GpuError::InternalError(msg.to_string())
    }
}
