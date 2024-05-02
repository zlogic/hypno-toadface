use std::ffi::c_void;
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
    descriptor_set: DescriptorSet,
    images: Vec<ImageBuffer>,
    param_buffers: Buffer,
    swapchain_extension: khr::swapchain::Device,
}

struct Display {
    width: u32,
    height: u32,
    surface: vk::SurfaceKHR,
}

struct Buffer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    mapped_memory: *mut c_void,
    stride: usize,
    count: usize,
}

struct Control {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    semaphore: vk::Semaphore,
}

struct DescriptorSet {
    descriptor_pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
    descriptor_set: vk::DescriptorSet,
}

struct Pipeline {
    vertex_shader_module: vk::ShaderModule,
    fragment_shader_module: vk::ShaderModule,
    //pipeline: vk::Pipeline,
    //layout: vk::PipelineLayout,
}

const IMAGE_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;
const MIN_IMAGES: u32 = 2;

struct ImageBuffer {
    fence: vk::Fence,
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
    command_buffer: vk::CommandBuffer,
}

struct ShaderParams {
    timecode: f32,
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

        let (physical_device, device_name, graphics_queue, min_ubo_alignment) =
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
        let pipeline = unsafe { Gpu::create_pipeline_layout(&device)? };
        let control = unsafe { Gpu::create_control(&device, graphics_queue)? };
        let images = unsafe {
            Gpu::create_image_buffers(
                &swapchain_extension,
                &device,
                swapchain,
                renderpass,
                control.command_pool,
                &display,
            )?
        };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let param_buffers = unsafe {
            Gpu::create_buffer(&device, &memory_properties, min_ubo_alignment, images.len())?
        };
        let descriptor_set = unsafe { Gpu::create_descriptor_set(&device, param_buffers.buffer)? };
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
            descriptor_set,
            images,
            param_buffers,
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
    ) -> Result<(vk::PhysicalDevice, String, u32, u64), GpuError> {
        let devices = instance.enumerate_physical_devices()?;
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = instance.get_physical_device_properties(device);
                if props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                // || props.limits.max_per_stage_descriptor_storage_buffers < MAX_BINDINGS
                // TODO: add pipeline/descriptor set limitations
                {
                    return None;
                }
                let queue_index = Gpu::find_graphics_queue(instance, device)?;
                let min_ubo_alignment = props.limits.min_uniform_buffer_offset_alignment;

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
                Some((device, device_name, queue_index, min_ubo_alignment, score))
            })
            .max_by(|(_, _, _, _, a), (_, _, _, _, b)| a.cmp(&b));
        let (device, name, queue_index, min_ubo_alignment) =
            if let Some((device, name, queue_index, min_ubo_alignment, _score)) = device {
                (device, name, queue_index, min_ubo_alignment)
            } else {
                return Err("Device not found".into());
            };
        Ok((device, name, queue_index, min_ubo_alignment))
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
                if queue
                    .queue_flags
                    .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER)
                    && queue.queue_count > 0
                {
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
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&enabled_extensions);
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode);

        Ok(swapchain_extension.create_swapchain(&swapchain_create_info, None)?)
    }

    unsafe fn create_image_buffers(
        swapchain_extension: &khr::swapchain::Device,
        device: &ash::Device,
        swapchain: vk::SwapchainKHR,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        display: &Display,
    ) -> Result<Vec<ImageBuffer>, vk::Result> {
        let swapchain_images = swapchain_extension.get_swapchain_images(swapchain)?;
        swapchain_images
            .iter()
            .map(|image| {
                Gpu::create_image_buffer(&device, *image, renderpass, command_pool, display)
            })
            .collect::<Result<Vec<_>, _>>()
    }

    unsafe fn create_image_buffer(
        device: &ash::Device,
        image: vk::Image,
        renderpass: vk::RenderPass,
        command_pool: vk::CommandPool,
        display: &Display,
    ) -> Result<ImageBuffer, vk::Result> {
        // TODO: cleanup on error
        let component_mapping = vk::ComponentMapping::default();
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

        let fence_create_info =
            vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = device.create_fence(&fence_create_info, None)?;
        let command_buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer = device.allocate_command_buffers(&command_buffers_info)?[0];
        Ok(ImageBuffer {
            fence,
            view,
            framebuffer,
            command_buffer,
        })
    }

    unsafe fn create_renderpass(device: &ash::Device) -> Result<vk::RenderPass, vk::Result> {
        // TODO: cleanup on error
        let attachment_description = vk::AttachmentDescription::default()
            .format(IMAGE_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let attachments = [attachment_description];
        let color_attachment = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachments = [color_attachment];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments)
            .resolve_attachments(&color_attachments);
        let subpasses = [subpass];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        device.create_render_pass(&render_pass_create_info, None)
    }

    unsafe fn create_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        min_ubo_alignment: u64,
        count: usize,
    ) -> Result<Buffer, GpuError> {
        let min_size = std::mem::size_of::<ShaderParams>() as u64;
        let min_size = if min_ubo_alignment == 0 || min_size % min_ubo_alignment == 0 {
            min_size
        } else {
            ((min_size / min_ubo_alignment) + 1) * min_ubo_alignment
        };
        let size = count as u64 * min_size;
        let required_memory_properties =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let usage_flags =
            vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER;
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
                if !memory_type
                    .property_flags
                    .contains(required_memory_properties)
                {
                    return None;
                }
                let allocate_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index as u32);
                // Some buffers may fill up, in this case allocating memory can fail.
                let mem = device.allocate_memory(&allocate_info, None).ok()?;

                Some(mem)
            })
            .next();

        let buffer_memory = if let Some(mem) = buffer_memory {
            mem
        } else {
            device.destroy_buffer(buffer, None);
            return Err("Cannot find suitable memory".into());
        };
        let mapped_memory = device
            .map_memory(buffer_memory, 0, size as u64, vk::MemoryMapFlags::empty())
            .map_err(|err| {
                device.destroy_buffer(buffer, None);
                device.free_memory(buffer_memory, None);
                err
            })?;
        // TODO: unmap memory on cleanup.
        Ok(Buffer {
            buffer,
            buffer_memory,
            mapped_memory,
            stride: min_size as usize,
            count,
        })
    }

    unsafe fn create_descriptor_set(
        device: &ash::Device,
        buffer: vk::Buffer,
    ) -> Result<DescriptorSet, vk::Result> {
        let descriptor_pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&descriptor_pool_size);
        // TODO: cleanup on error
        let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;

        let layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);
        let layout_bindings = [layout_binding];
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
        let layout = device.create_descriptor_set_layout(&layout_info, None)?;

        let layouts = [layout];

        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = device.allocate_descriptor_sets(&descriptor_set_allocate_info)?;
        let descriptor_set = descriptor_sets[0];

        let buffer_info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(std::mem::size_of::<ShaderParams>() as u64);
        let buffer_infos = [buffer_info];
        let write_descriptor = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC)
            .buffer_info(&buffer_infos);
        device.update_descriptor_sets(&[write_descriptor], &[]);
        Ok(DescriptorSet {
            descriptor_pool,
            layout,
            descriptor_set,
        })
    }

    unsafe fn create_pipeline_layout(device: &ash::Device) -> Result<Pipeline, GpuError> {
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
            .expect("Vertex shader module error");
        let fragment_shader_module = device
            .create_shader_module(&fragment_shader_info, None)
            .expect("Fragment shader module error");

        let layout_create_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = device
            .create_pipeline_layout(&layout_create_info, None)
            .unwrap();

        let shader_entry_name = c"main";
        let shader_stage_create_infos = [
            vk::PipelineShaderStageCreateInfo::default()
                .module(vertex_shader_module)
                .name(shader_entry_name)
                .stage(vk::ShaderStageFlags::VERTEX),
            vk::PipelineShaderStageCreateInfo::default()
                .module(fragment_shader_module)
                .name(shader_entry_name)
                .stage(vk::ShaderStageFlags::FRAGMENT),
        ];

        /*
        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<Vertex>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Vertex, color) as u32,
            },
        ];
        */

        Ok(Pipeline {
            vertex_shader_module,
            fragment_shader_module,
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
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = device
            .create_fence(&fence_create_info, None)
            .map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_command_pool(command_pool, None);
            device.destroy_fence(fence, None);
            err
        };
        let semaphore_create_info = vk::SemaphoreCreateInfo::default();
        let semaphore = device
            .create_semaphore(&semaphore_create_info, None)
            .map_err(cleanup_err)?;
        Ok(Control {
            queue,
            command_pool,
            fence,
            semaphore,
        })
    }

    unsafe fn copy_buffer_data(
        &self,
        image_index: usize,
        image: &ImageBuffer,
        scene: &graphics::Scene,
    ) {
        // TODO: panic if index is out of range?
        let buffer = &self.param_buffers;
        let offset = (buffer.stride * image_index) as isize;
        let dst = buffer.mapped_memory.offset(offset);
        let params = ShaderParams {
            timecode: scene.timecode as f32,
        };
        (dst as *mut ShaderParams).write(params);

        /*
        self.device.cmd_bind_descriptor_sets(
            image.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.layout,
            0,
            &[self.descriptor_set.descriptor_set],
            &[offset as u32],
        );
        */
    }

    unsafe fn render_image(
        &self,
        scene: &graphics::Scene,
        image: &ImageBuffer,
    ) -> Result<(), vk::Result> {
        let (r, g, b) = {
            let timecode = scene.timecode / 10.0;
            let timecode = timecode - timecode.floor() as f64;
            let h = timecode;
            let v = 1.0;
            let s = 1.0;
            let i = (h * 6.0).floor() as u8;
            let f = h * 6.0 - i as f64;
            let p = v * (1.0 - s);
            let q = v * (1.0 - f * s);
            let t = v * (1.0 - (1.0 - f) * s);
            match i % 6 {
                0 => (v, t, p),
                1 => (q, v, p),
                2 => (p, v, t),
                3 => (p, q, v),
                4 => (t, p, v),
                5 => (v, p, q),
                _ => (0.0, 0.0, 0.0),
            }
        };
        /*
        let colorvalue = (scene.i % 2) as f32;
        let (r, g, b) = (colorvalue, colorvalue, colorvalue);
        */

        let command_buffer = image.command_buffer;

        self.device
            .wait_for_fences(&[image.fence], true, u64::MAX)?;
        self.device.reset_fences(&[image.fence])?;
        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [r as f32, g as f32, b as f32, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
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
        self.device.cmd_end_render_pass(command_buffer);

        self.device.end_command_buffer(command_buffer)?;
        let wait_semaphores = [self.control.semaphore];
        let command_buffers = [command_buffer];
        let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let queue_submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .command_buffers(&command_buffers)
            .wait_dst_stage_mask(&wait_dst_stage_mask);

        self.device
            .queue_submit(self.control.queue, &[queue_submit_info], image.fence)
    }

    pub fn render(&self, scene: &graphics::Scene) -> Result<RenderFeedback, GpuError> {
        unsafe {
            let (image_index, swapchain_suboptimal) = self.swapchain_extension.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.control.semaphore,
                self.control.fence,
            )?;

            let image = &self.images[image_index as usize];
            self.copy_buffer_data(image_index as usize, image, scene);
            self.render_image(scene, image)?;

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let queue_present_info = vk::PresentInfoKHR::default()
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let queue_suboptimal = self
                .swapchain_extension
                .queue_present(self.control.queue, &queue_present_info)?;
            self.device.queue_wait_idle(self.control.queue)?;
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
    VkError(vk::Result),
    IoError(io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GpuError::InternalError(ref msg) => f.write_str(msg),
            GpuError::LoadingError(ref e) => {
                write!(f, "Failed to init GPU: {}", e)
            }
            GpuError::VkError(ref e) => {
                write!(f, "Vulkan error: {}", e)
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
            GpuError::VkError(ref e) => Some(e),
            GpuError::IoError(ref e) => Some(e),
        }
    }
}

impl From<ash::LoadingError> for GpuError {
    fn from(err: ash::LoadingError) -> GpuError {
        GpuError::LoadingError(err)
    }
}

impl From<ash::vk::Result> for GpuError {
    fn from(err: ash::vk::Result) -> GpuError {
        GpuError::VkError(err)
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
