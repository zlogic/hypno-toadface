use std::{error, fmt, slice};

use ash::prelude::VkResult;
use ash::vk;

use crate::display::SurfaceParameters;
use crate::graphics;

struct Device {
    instance: ash::Instance,
}

pub struct Gpu {
    surface: SurfaceParameters,
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    device_name: String,
    image: Image,
    framebuffer: Framebuffer,
    renderpass: vk::RenderPass,
    control: Control,
}

struct Control {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

struct Image {
    image: vk::Image,
    memory: vk::DeviceMemory,
    size_bytes: u64,
    format: vk::Format,
}

struct Framebuffer {
    view: vk::ImageView,
    framebuffer: vk::Framebuffer,
}

struct ShaderParams {
    timecode: f32,
    center_x: u32,
    center_y: u32,
}

impl Gpu {
    pub fn init(parameters: SurfaceParameters) -> Result<Gpu, GpuError> {
        let entry = unsafe { ash::Entry::load()? };
        let instance = unsafe { Gpu::init_vk(&entry)? };

        // ash_window would have supported DRM, but needs Vulkan extensions to make it work.
        let max_dimensions = parameters.width.max(parameters.height);
        let (physical_device, device_name, graphics_queue) =
            unsafe { Gpu::find_device(&instance, max_dimensions as u32)? };

        let device = unsafe { Gpu::create_device(&instance, physical_device, graphics_queue)? };
        let image = unsafe {
            let memory_properties = instance.get_physical_device_memory_properties(physical_device);
            Gpu::create_image(
                &device,
                &memory_properties,
                parameters.stride,
                parameters.height as u32,
            )?
        };
        let renderpass = unsafe { Gpu::create_renderpass(&device, &image)? };
        let framebuffer = unsafe {
            Gpu::create_framebuffer(
                &device,
                &image,
                renderpass,
                parameters.width as u32,
                parameters.height as u32,
            )?
        };
        // TODO: cleanup on error
        let control = unsafe { Gpu::create_control(&device, graphics_queue)? };
        Ok(Gpu {
            surface: parameters,
            entry,
            instance,
            device,
            device_name,
            image,
            framebuffer,
            renderpass,
            control,
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

        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(create_flags);
        entry.create_instance(&create_info, None)
    }

    unsafe fn find_device(
        instance: &ash::Instance,
        max_dimension: u32,
    ) -> Result<(vk::PhysicalDevice, String, u32), GpuError> {
        let devices = instance.enumerate_physical_devices()?;
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = instance.get_physical_device_properties(device);
                if props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                    || props.limits.max_image_dimension2_d < max_dimension
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

    unsafe fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        graphics_queue_index: u32,
    ) -> Result<ash::Device, vk::Result> {
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_index)
            .queue_priorities(&[1.0f32]);
        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        instance.create_device(physical_device, &device_create_info, None)
    }

    unsafe fn create_image(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        stride: u32,
        height: u32,
    ) -> Result<Image, GpuError> {
        // Match the DRM buffer format.
        let format = vk::Format::B8G8R8A8_UNORM;
        // TODO: cleanup on error
        let extent = vk::Extent3D::default()
            .width(stride)
            .height(height)
            .depth(1);
        let image_create_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::TRANSFER_DST)
            .extent(extent)
            .array_layers(1)
            .mip_levels(1)
            .samples(vk::SampleCountFlags::TYPE_1);

        let image = device.create_image(&image_create_info, None)?;

        // TODO: VK_EXT_external_memory_dma_buf or VK_EXT_external_memory_host might allow to share memory.
        let memory_requirements = device.get_image_memory_requirements(image);

        let required_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
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
                if !memory_type.property_flags.contains(required_flags) {
                    return None;
                }
                let allocate_info = vk::MemoryAllocateInfo {
                    allocation_size: memory_requirements.size,
                    memory_type_index: memory_type_index as u32,
                    ..Default::default()
                };
                // Some buffers may fill up, in this case allocating memory can fail.
                let mem = device.allocate_memory(&allocate_info, None).ok()?;

                Some(mem)
            })
            .next();
        let memory = buffer_memory.ok_or("Cannot find suitable memory")?;

        device.bind_image_memory(image, memory, 0)?;

        Ok(Image {
            image,
            memory,
            size_bytes: memory_requirements.size,
            format,
        })
    }

    unsafe fn create_renderpass(
        device: &ash::Device,
        image: &Image,
    ) -> Result<vk::RenderPass, vk::Result> {
        // TODO: cleanup on error
        let attachment_description = vk::AttachmentDescription::default()
            .format(image.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::GENERAL);
        let attachments = [attachment_description];
        let color_attachment = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::GENERAL);
        let color_attachments = [color_attachment];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachments);
        let subpasses = [subpass];
        let render_pass_create_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses);
        device.create_render_pass(&render_pass_create_info, None)
    }

    unsafe fn create_framebuffer(
        device: &ash::Device,
        image: &Image,
        render_pass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> Result<Framebuffer, vk::Result> {
        // TODO: cleanup on error

        // Since the only image is the render target, attach it to a view and framebuffer.
        let component_mapping = vk::ComponentMapping::default();
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_info = vk::ImageViewCreateInfo::default()
            .image(image.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(image.format)
            .components(component_mapping)
            .subresource_range(subresource_range);
        let view = device.create_image_view(&image_view_info, None)?;

        let attachments = [view];
        let framebuffer_create_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&attachments)
            .width(width)
            .height(height)
            .layers(1);
        let framebuffer = device.create_framebuffer(&framebuffer_create_info, None)?;

        Ok(Framebuffer { view, framebuffer })
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
        let command_buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer = device
            .allocate_command_buffers(&command_buffers_info)
            .map_err(cleanup_err)?[0];
        Ok(Control {
            queue,
            command_pool,
            fence,
            command_buffer,
        })
    }

    pub unsafe fn render(
        &self,
        scene: &graphics::Scene,
        dst_buffer: &mut [u8],
    ) -> Result<(), GpuError> {
        let command_buffer = self.control.command_buffer;

        self.device.reset_fences(&[self.control.fence])?;
        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;
        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [255.0, 255.0, 255.0, 255.0],
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
                width: self.surface.width as u32,
                height: self.surface.height as u32,
            },
        };

        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(self.renderpass)
            .framebuffer(self.framebuffer.framebuffer)
            .render_area(render_area)
            .clear_values(&clear_values);

        self.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin,
            vk::SubpassContents::INLINE,
        );
        self.device.cmd_end_render_pass(command_buffer);

        // TODO: copy image here
        self.device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        self.device
            .queue_submit(self.control.queue, &[submit_info], self.control.fence)?;
        self.device
            .wait_for_fences(&[self.control.fence], true, u64::MAX)?;

        let mapped_memory = self.device.map_memory(
            self.image.memory,
            0,
            self.image.size_bytes,
            vk::MemoryMapFlags::empty(),
        )?;
        {
            let size = self.image.size_bytes as usize;
            let mapped_slice = slice::from_raw_parts(mapped_memory as *const u8, size);
            dst_buffer.copy_from_slice(mapped_slice);
        }
        self.device.unmap_memory(self.image.memory);

        Ok(())
    }
}

#[derive(Debug)]
pub enum GpuError {
    InternalError(String),
    LoadingError(ash::LoadingError),
    VkError(vk::Result),
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
        }
    }
}

impl error::Error for GpuError {
    fn cause(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            GpuError::InternalError(ref _msg) => None,
            GpuError::LoadingError(ref e) => Some(e),
            GpuError::VkError(ref e) => Some(e),
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

impl From<&str> for GpuError {
    fn from(msg: &str) -> GpuError {
        GpuError::InternalError(msg.to_string())
    }
}
