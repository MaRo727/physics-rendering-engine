use std::ffi::{c_char, CStr, CString};

use anyhow::{bail, Context, Result};
use ash::{vk, Device, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

#[cfg(debug_assertions)]
const VALIDATION_LAYERS: &[*const c_char] = &[c"VK_LAYER_KHRONOS_validation".as_ptr()];

const DEVICE_EXTENSIONS: &[*const c_char] = &[ash::khr::swapchain::NAME.as_ptr()];

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_loader: ash::khr::surface::Instance,
    pub surface: vk::SurfaceKHR,
    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub graphics_queue_family: u32,
    #[cfg(debug_assertions)]
    debug_utils_loader: ash::ext::debug_utils::Instance,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl VulkanContext {
    pub fn new(window: &(impl HasDisplayHandle + HasWindowHandle)) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("Failed to load Vulkan — is a driver installed?")?;

        let instance = create_instance(&entry, window)?;

        #[cfg(debug_assertions)]
        let (debug_utils_loader, debug_messenger) = setup_debug_messenger(&entry, &instance)?;

        let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }
        .context("Failed to create Vulkan surface")?;

        let (physical_device, graphics_queue_family) =
            pick_physical_device(&instance, &surface_loader, surface)?;

        let device =
            create_logical_device(&instance, physical_device, graphics_queue_family)?;

        let graphics_queue =
            unsafe { device.get_device_queue(graphics_queue_family, 0) };

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            graphics_queue,
            graphics_queue_family,
            #[cfg(debug_assertions)]
            debug_utils_loader,
            #[cfg(debug_assertions)]
            debug_messenger,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_device(None);
            #[cfg(debug_assertions)]
            self.debug_utils_loader
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// ---------------------------------------------------------------------------
// Instance creation
// ---------------------------------------------------------------------------

fn create_instance(
    entry: &Entry,
    window: &(impl HasDisplayHandle + HasWindowHandle),
) -> Result<Instance> {
    let app_name = CString::new("Physics Rendering Engine").unwrap();
    let engine_name = CString::new("No Engine").unwrap();

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(&engine_name)
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_3);

    let display_handle = window.display_handle()?;
    let mut extensions =
        ash_window::enumerate_required_extensions(display_handle.as_raw())
            .context("Failed to enumerate required instance extensions")?
            .to_vec();

    #[cfg(debug_assertions)]
    {
        check_validation_layers(entry)?;
        extensions.push(ash::ext::debug_utils::NAME.as_ptr());
    }

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extensions);

    #[cfg(debug_assertions)]
    let create_info = create_info.enabled_layer_names(VALIDATION_LAYERS);

    let instance = unsafe { entry.create_instance(&create_info, None) }
        .context("Failed to create Vulkan instance")?;

    Ok(instance)
}

// ---------------------------------------------------------------------------
// Validation layers
// ---------------------------------------------------------------------------

#[cfg(debug_assertions)]
fn check_validation_layers(entry: &Entry) -> Result<()> {
    let available = unsafe { entry.enumerate_instance_layer_properties() }
        .context("Failed to enumerate instance layer properties")?;

    for &required in VALIDATION_LAYERS {
        let required = unsafe { CStr::from_ptr(required) };
        let found = available.iter().any(|layer| {
            let name = unsafe { CStr::from_ptr(layer.layer_name.as_ptr()) };
            name == required
        });
        if !found {
            bail!(
                "Validation layer {:?} not available — install the Vulkan SDK",
                required
            );
        }
    }
    Ok(())
}

#[cfg(debug_assertions)]
fn setup_debug_messenger(
    entry: &Entry,
    instance: &Instance,
) -> Result<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)> {
    let loader = ash::ext::debug_utils::Instance::new(entry, instance);

    let create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None) }
        .context("Failed to create debug messenger")?;

    Ok((loader, messenger))
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let message = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan] {message}");
    } else {
        log::info!("[Vulkan] {message}");
    }
    vk::FALSE
}

// ---------------------------------------------------------------------------
// Physical device selection
// ---------------------------------------------------------------------------

fn pick_physical_device(
    instance: &Instance,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let devices = unsafe { instance.enumerate_physical_devices() }
        .context("Failed to enumerate physical devices")?;

    if devices.is_empty() {
        bail!("No Vulkan-capable GPU found");
    }

    let mut fallback: Option<(vk::PhysicalDevice, u32)> = None;

    for &device in &devices {
        let Some(queue_family) =
            find_graphics_queue_family(instance, device, surface_loader, surface)
        else {
            continue;
        };
        if !device_supports_extensions(instance, device) {
            continue;
        }

        let props = unsafe { instance.get_physical_device_properties(device) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };

        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            log::info!("Selected GPU: {name:?} (discrete)");
            return Ok((device, queue_family));
        }
        fallback.get_or_insert((device, queue_family));
    }

    fallback
        .map(|(device, family)| {
            let props = unsafe { instance.get_physical_device_properties(device) };
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) };
            log::info!("Selected GPU: {name:?} (fallback)");
            (device, family)
        })
        .context("No suitable GPU found (needs graphics + present queue, swapchain extension)")
}

fn find_graphics_queue_family(
    instance: &Instance,
    device: vk::PhysicalDevice,
    surface_loader: &ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
) -> Option<u32> {
    let families =
        unsafe { instance.get_physical_device_queue_family_properties(device) };

    families.iter().enumerate().find_map(|(i, family)| {
        let i = i as u32;
        let has_graphics = family.queue_flags.contains(vk::QueueFlags::GRAPHICS);
        let has_present = unsafe {
            surface_loader
                .get_physical_device_surface_support(device, i, surface)
                .unwrap_or(false)
        };
        (has_graphics && has_present).then_some(i)
    })
}

fn device_supports_extensions(instance: &Instance, device: vk::PhysicalDevice) -> bool {
    let available =
        unsafe { instance.enumerate_device_extension_properties(device) }
            .unwrap_or_default();

    DEVICE_EXTENSIONS.iter().all(|&required| {
        let required = unsafe { CStr::from_ptr(required) };
        available
            .iter()
            .any(|ext| unsafe { CStr::from_ptr(ext.extension_name.as_ptr()) } == required)
    })
}

// ---------------------------------------------------------------------------
// Logical device creation
// ---------------------------------------------------------------------------

fn create_logical_device(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<Device> {
    let queue_priority = 1.0_f32;
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(std::slice::from_ref(&queue_priority));

    let features = vk::PhysicalDeviceFeatures::default();

    let create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info))
        .enabled_extension_names(DEVICE_EXTENSIONS)
        .enabled_features(&features);

    // Note: device-level layers were deprecated in Vulkan 1.0; validation is
    // controlled entirely by the instance layer set up in create_instance().

    let device =
        unsafe { instance.create_device(physical_device, &create_info, None) }
            .context("Failed to create logical device")?;

    Ok(device)
}
