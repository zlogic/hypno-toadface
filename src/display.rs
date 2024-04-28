use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::io::{AsFd, BorrowedFd};
use std::time::Instant;
use std::{error, fmt, io};

use drm::buffer::{Buffer, DrmFourcc};
use drm::control;
use drm::control::atomic;
use drm::control::connector;
use drm::control::crtc;
use drm::control::dumbbuffer;
use drm::control::framebuffer;
use drm::control::property;
use drm::control::AtomicCommitFlags;
use drm::control::PageFlipFlags;

use drm::{control::Device as _, Device as _};

use crate::graphics::{Renderer, Scene};

struct Card(File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl drm::Device for Card {}
impl drm::control::Device for Card {}

pub struct Surface {
    card: Card,
    front_buffer: CardBuffer,
    back_buffer: CardBuffer,
    device_name: String,
    parameters: SurfaceParameters,
    crtc: control::crtc::Handle,
}

struct CardBuffer {
    fb: framebuffer::Handle,
    db: dumbbuffer::DumbBuffer,
}

#[derive(Clone, Copy)]
pub struct SurfaceParameters {
    pub width: u16,
    pub height: u16,
    pub stride: u32,
}

impl Surface {
    pub fn open(path: &str) -> Result<Self, SurfaceError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|err| ("Failed to open DRM file", err))?;
        let card = Card(file);

        card.set_client_capability(drm::ClientCapability::UniversalPlanes, true)
            .map_err(|err| ("Unable to request UniversalPlanes capability", err))?;
        card.set_client_capability(drm::ClientCapability::Atomic, true)
            .map_err(|err| ("Unable to request Atomic capability", err))?;

        // Load the information.
        let res = card.resource_handles()?;
        let coninfo: Vec<connector::Info> = res
            .connectors()
            .iter()
            .flat_map(|con| card.get_connector(*con, true))
            .collect();
        let crtcinfo: Vec<crtc::Info> = res
            .crtcs()
            .iter()
            .flat_map(|crtc| card.get_crtc(*crtc))
            .collect();

        // Filter each connector until we find one that's connected.
        let con = coninfo
            .iter()
            .find(|&i| i.state() == connector::State::Connected)
            .ok_or("No connected connectors")?;

        // Get the first (usually best) mode.
        let &mode = con.modes().first().ok_or("No modes found on connector")?;

        let (disp_width, disp_height) = mode.size();

        // Find a crtc and FB.
        let crtc = crtcinfo.first().ok_or("No crtcs found")?;

        // Select the pixel format.
        let fmt = DrmFourcc::Xrgb8888;

        // Create buffers.
        let create_buffer = || -> Result<CardBuffer, SurfaceError> {
            let db = card
                .create_dumb_buffer((disp_width.into(), disp_height.into()), fmt, 32)
                .map_err(|err| ("Could not create dumb buffer", err))?;
            let fb = card
                .add_framebuffer(&db, 24, 32)
                .map_err(|err| ("Could not create FB", err))?;
            let buf = CardBuffer { db, fb };
            Ok(buf)
        };
        let front_buffer: CardBuffer = create_buffer()?;
        let back_buffer: CardBuffer = create_buffer()?;
        let pitch = front_buffer.db.pitch();
        let planes = card
            .plane_handles()
            .map_err(|err| ("Could not list planes", err))?;
        let (better_planes, compatible_planes): (
            Vec<control::plane::Handle>,
            Vec<control::plane::Handle>,
        ) = planes
            .iter()
            .filter(|&&plane| {
                card.get_plane(plane)
                    .map(|plane_info| {
                        let compatible_crtcs = res.filter_crtcs(plane_info.possible_crtcs());
                        compatible_crtcs.contains(&crtc.handle())
                    })
                    .unwrap_or(false)
            })
            .partition(|&&plane| {
                if let Ok(props) = card.get_properties(plane) {
                    for (&id, &val) in props.iter() {
                        if let Ok(info) = card.get_property(id) {
                            if info.name().to_str().map(|x| x == "type").unwrap_or(false) {
                                return val == (drm::control::PlaneType::Primary as u32).into();
                            }
                        }
                    }
                }
                false
            });
        let plane = *better_planes.first().unwrap_or(&compatible_planes[0]);

        let con_props = card
            .get_properties(con.handle())
            .map_err(|err| ("Could not get props of ronnector", err))?
            .as_hashmap(&card)
            .map_err(|err| ("Could not get a prop from connector", err))?;
        let crtc_props = card
            .get_properties(crtc.handle())
            .map_err(|err| ("Could not get props of crtc", err))?
            .as_hashmap(&card)
            .map_err(|err| ("Could not get a prop from crtc", err))?;
        let plane_props = card
            .get_properties(plane)
            .map_err(|err| ("Could not get props of plane", err))?
            .as_hashmap(&card)
            .map_err(|err| ("Could not get a prop from plane", err))?;

        // Set initial properties.
        let mut atomic_req = atomic::AtomicModeReq::new();
        atomic_req.add_property(
            con.handle(),
            con_props["CRTC_ID"].handle(),
            property::Value::CRTC(Some(crtc.handle())),
        );
        let blob = card
            .create_property_blob(&mode)
            .map_err(|err| ("Failed to create blob", err))?;
        atomic_req.add_property(crtc.handle(), crtc_props["MODE_ID"].handle(), blob);
        atomic_req.add_property(
            crtc.handle(),
            crtc_props["ACTIVE"].handle(),
            property::Value::Boolean(true),
        );
        atomic_req.add_property(
            plane,
            plane_props["FB_ID"].handle(),
            property::Value::Framebuffer(Some(front_buffer.fb)),
        );
        atomic_req.add_property(
            plane,
            plane_props["CRTC_ID"].handle(),
            property::Value::CRTC(Some(crtc.handle())),
        );
        atomic_req.add_property(
            plane,
            plane_props["SRC_X"].handle(),
            property::Value::UnsignedRange(0),
        );
        atomic_req.add_property(
            plane,
            plane_props["SRC_Y"].handle(),
            property::Value::UnsignedRange(0),
        );
        atomic_req.add_property(
            plane,
            plane_props["SRC_W"].handle(),
            property::Value::UnsignedRange((mode.size().0 as u64) << 16),
        );
        atomic_req.add_property(
            plane,
            plane_props["SRC_H"].handle(),
            property::Value::UnsignedRange((mode.size().1 as u64) << 16),
        );
        atomic_req.add_property(
            plane,
            plane_props["CRTC_X"].handle(),
            property::Value::SignedRange(0),
        );
        atomic_req.add_property(
            plane,
            plane_props["CRTC_Y"].handle(),
            property::Value::SignedRange(0),
        );
        atomic_req.add_property(
            plane,
            plane_props["CRTC_W"].handle(),
            property::Value::UnsignedRange(mode.size().0 as u64),
        );
        atomic_req.add_property(
            plane,
            plane_props["CRTC_H"].handle(),
            property::Value::UnsignedRange(mode.size().1 as u64),
        );

        // Set the crtc
        // On many setups, this requires root access.
        card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, atomic_req)
            .map_err(|e| ("Failed to set mode", e))?;

        let device_name = card.get_driver()?.name().to_str().unwrap().to_string();
        let parameters = SurfaceParameters {
            width: disp_width,
            height: disp_height,
            // Pitch contains bytes, not pixels.
            stride: pitch / 4,
        };

        let surface = Self {
            card,
            front_buffer,
            back_buffer,
            device_name,
            parameters,
            crtc: crtc.handle(),
        };
        Ok(surface)
    }

    pub fn parameters(&self) -> SurfaceParameters {
        self.parameters
    }

    pub fn render_loop(&mut self, mut renderer: Box<dyn Renderer>) -> Result<(), SurfaceError> {
        let mut framecounter_start = Instant::now();
        let start = Instant::now();
        let mut framecounter_frames = 0usize;

        let mut front_mapping = self
            .card
            .map_dumb_buffer(&mut self.front_buffer.db)
            .map_err(|err| ("Could not map dumbbuffer", err))?;
        let mut back_mapping = self
            .card
            .map_dumb_buffer(&mut self.back_buffer.db)
            .map_err(|err| ("Could not map dumbbuffer", err))?;
        // TODO: Refactor this to be more reusable
        self.card
            .page_flip(self.crtc, self.front_buffer.fb, PageFlipFlags::EVENT, None)?;
        let _ = self.card.receive_events();
        let mut display_front = true;
        loop {
            display_front = !display_front;
            let scene = Scene {
                timecode: start.elapsed().as_secs_f64() * 4.0,
            };
            if display_front {
                match renderer.render(&scene, front_mapping.as_mut()) {
                    Ok(_) => {}
                    Err(err) => eprintln!("{}", err),
                }
            } else {
                match renderer.render(&scene, back_mapping.as_mut()) {
                    Ok(_) => {}
                    Err(err) => eprintln!("{}", err),
                }
            }

            let show_fb = if display_front {
                self.front_buffer.fb
            } else {
                self.back_buffer.fb
            };
            self.card
                .page_flip(self.crtc, show_fb, PageFlipFlags::EVENT, None)?;
            let _ = self.card.receive_events();

            framecounter_frames += 1;
            if framecounter_frames > 30 {
                let elapsed = framecounter_start.elapsed();
                let fps = framecounter_frames as f32 / elapsed.as_secs_f32() as f32;

                print!("Average FPS: {:.2}   \r", fps);
                let _ = io::stdout().flush();

                framecounter_frames = 0;
                framecounter_start = Instant::now();
            }
        }
    }

    pub fn device_name(&self) -> &str {
        self.device_name.as_str()
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        let _ = self.card.destroy_framebuffer(self.front_buffer.fb);
        let _ = self.card.destroy_framebuffer(self.back_buffer.fb);
        let _ = self.card.destroy_dumb_buffer(self.front_buffer.db);
        let _ = self.card.destroy_dumb_buffer(self.back_buffer.db);
    }
}

#[derive(Debug)]
pub enum SurfaceError {
    InternalError(String),
    IoError(String, io::Error),
}

impl fmt::Display for SurfaceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SurfaceError::InternalError(ref msg) => f.write_str(msg),
            SurfaceError::IoError(ref msg, ref e) => {
                if !msg.is_empty() {
                    write!(f, "{}: {}", msg, e)
                } else {
                    e.fmt(f)
                }
            }
        }
    }
}

impl error::Error for SurfaceError {
    fn cause(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SurfaceError::InternalError(ref _msg) => None,
            SurfaceError::IoError(ref _msg, ref e) => Some(e),
        }
    }
}

impl From<(&str, io::Error)> for SurfaceError {
    fn from(e: (&str, io::Error)) -> SurfaceError {
        SurfaceError::IoError(e.0.to_string(), e.1)
    }
}

impl From<io::Error> for SurfaceError {
    fn from(err: io::Error) -> SurfaceError {
        SurfaceError::IoError(String::new(), err)
    }
}

impl From<&str> for SurfaceError {
    fn from(msg: &str) -> SurfaceError {
        SurfaceError::InternalError(msg.to_string())
    }
}
