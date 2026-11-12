use std::ffi::{CStr, c_char, c_void};
use std::sync::atomic::{AtomicU32, Ordering};

use ash::{ext, khr, vk};
use raw_window_handle::RawDisplayHandle;

use crate::error::{Result, VkResultExt};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// What surfaces the instance must be able to create.
#[derive(Clone, Copy, Debug)]
pub enum SurfaceSupport {
    /// Offscreen only (tools and tests).
    None,
    /// Windows on this display.
    Display(RawDisplayHandle),
    /// `VK_EXT_headless_surface`: a swapchain with no window, for tests.
    Headless,
}

/// A Vulkan instance with its loader, optional validation and surface extensions.
pub struct Instance {
    pub(crate) entry: ash::Entry,
    pub(crate) raw: ash::Instance,
    pub(crate) surface: Option<khr::surface::Instance>,
    pub(crate) headless: Option<ext::headless_surface::Instance>,
    debug: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    validation_errors: Box<AtomicU32>,
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    kind: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user: *mut c_void,
) -> vk::Bool32 {
    // SAFETY: the loader passes a valid callback data struct whose message is a NUL-terminated
    // string, and `user` is the boxed counter that outlives the messenger.
    let (message, errors) = unsafe {
        let message = (*data)
            .message_as_c_str()
            .map(CStr::to_string_lossy)
            .unwrap_or_default();
        (message, &*(user as *const AtomicU32))
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        errors.fetch_add(1, Ordering::Relaxed);
        tracing::error!(target: "vulkan", ?kind, "{message}");
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        tracing::warn!(target: "vulkan", ?kind, "{message}");
    } else {
        tracing::debug!(target: "vulkan", ?kind, "{message}");
    }
    vk::FALSE
}

impl Instance {
    /// Loads Vulkan and creates an instance. Validation is enabled when `validation` is set
    /// and the Khronos layer is installed; otherwise it is skipped with a warning.
    pub fn new(validation: bool, surfaces: SurfaceSupport) -> Result<Self> {
        // SAFETY: loading the system Vulkan library runs its initialisation code, which is
        // the documented way to use the loader.
        let entry = unsafe { ash::Entry::load()? };
        // SAFETY: plain enumeration calls on a valid entry.
        let (layers, extensions) = unsafe {
            (
                entry
                    .enumerate_instance_layer_properties()
                    .call("vkEnumerateInstanceLayerProperties")?,
                entry
                    .enumerate_instance_extension_properties(None)
                    .call("vkEnumerateInstanceExtensionProperties")?,
            )
        };
        let has_extension = |name: &CStr| {
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let has_layer = layers
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(VALIDATION_LAYER));
        let validation = validation && has_layer && has_extension(ext::debug_utils::NAME);
        if !validation && has_layer {
            tracing::debug!("validation layers off");
        }

        let mut enabled: Vec<*const c_char> = Vec::new();
        let mut want_surface = false;
        let mut want_headless = false;
        match surfaces {
            SurfaceSupport::None => {}
            SurfaceSupport::Display(display) => {
                enabled.extend_from_slice(
                    ash_window::enumerate_required_extensions(display)
                        .call("surface extensions")?,
                );
                want_surface = true;
            }
            SurfaceSupport::Headless => {
                if has_extension(ext::headless_surface::NAME) {
                    enabled.push(khr::surface::NAME.as_ptr());
                    enabled.push(ext::headless_surface::NAME.as_ptr());
                    want_surface = true;
                    want_headless = true;
                }
            }
        }
        if validation {
            enabled.push(ext::debug_utils::NAME.as_ptr());
        }
        let portability = has_extension(khr::portability_enumeration::NAME);
        if portability {
            enabled.push(khr::portability_enumeration::NAME.as_ptr());
        }
        let layer_names = [VALIDATION_LAYER.as_ptr()];
        let app = vk::ApplicationInfo::default()
            .application_name(c"Lichii")
            .engine_name(c"Lichii")
            .api_version(vk::API_VERSION_1_3);
        let mut info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&enabled);
        if validation {
            info = info.enabled_layer_names(&layer_names);
        }
        if portability {
            info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }
        // SAFETY: every pointer in `info` points to data that lives until the call returns.
        let raw = unsafe {
            entry
                .create_instance(&info, None)
                .call("vkCreateInstance")?
        };

        let validation_errors = Box::new(AtomicU32::new(0));
        let debug = validation.then(|| {
            let loader = ext::debug_utils::Instance::new(&entry, &raw);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback))
                .user_data(&*validation_errors as *const AtomicU32 as *mut c_void);
            // SAFETY: the callback and its user data (boxed, owned by self) outlive the
            // messenger, which is destroyed first in Drop.
            unsafe { loader.create_debug_utils_messenger(&info, None) }
                .ok()
                .map(|messenger| (loader, messenger))
        });
        let surface = want_surface.then(|| khr::surface::Instance::new(&entry, &raw));
        let headless = want_headless.then(|| ext::headless_surface::Instance::new(&entry, &raw));
        Ok(Self {
            entry,
            raw,
            surface,
            headless,
            debug: debug.flatten(),
            validation_errors,
        })
    }

    pub fn raw(&self) -> &ash::Instance {
        &self.raw
    }

    /// Whether the validation layer is active.
    pub fn validation_enabled(&self) -> bool {
        self.debug.is_some()
    }

    /// Validation errors reported so far (0 without validation).
    pub fn validation_errors(&self) -> u32 {
        self.validation_errors.load(Ordering::Relaxed)
    }

    /// Whether a headless test surface can be created.
    pub fn supports_headless_surface(&self) -> bool {
        self.headless.is_some()
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: every child object (devices, surfaces) holds an Arc to this instance and was
        // destroyed before it; the messenger is destroyed before the instance.
        unsafe {
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.raw.destroy_instance(None);
        }
    }
}
