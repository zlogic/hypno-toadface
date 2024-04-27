use std::fs::{File, OpenOptions};
use std::os::unix::io::{AsFd, BorrowedFd};
use std::{error, fmt, io};

use drm::control::{dumbbuffer, framebuffer, Device as _, PageFlipFlags};
use drm::Device as _;

use drm::buffer::DrmFourcc;
use drm::control;
use drm::control::atomic;
use drm::control::connector;
use drm::control::crtc;
use drm::control::property;
use drm::control::AtomicCommitFlags;

struct Card(File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl drm::Device for Card {}
impl drm::control::Device for Card {}

impl Card {
    pub fn open(path: &str) -> Result<Self, io::Error> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Card(file))
    }
}

pub struct Surface {
    card: Card,
    buffers: Vec<CardBuffer>,
    show_buffer: usize,
    write_buffer: usize,
    crtc: control::crtc::Handle,
}

struct CardBuffer {
    width: u16,
    height: u16,
    fb: framebuffer::Handle,
    db: dumbbuffer::DumbBuffer,
}

pub struct WritableSurface<'a> {
    width: u16,
    height: u16,
    mapping: dumbbuffer::DumbMapping<'a>,
}

impl Surface {
    pub fn open(path: &str) -> Result<Self, SurfaceError> {
        let card = Card::open(path)?;

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
        let buffers = (0..2)
            .map(|_| {
                let db = card
                    .create_dumb_buffer((disp_width.into(), disp_height.into()), fmt, 32)
                    .map_err(|err| ("Could not create dumb buffer", err))?;
                let fb = card
                    .add_framebuffer(&db, 24, 32)
                    .map_err(|err| ("Could not create FB", err))?;
                let buf = CardBuffer {
                    db,
                    fb,
                    width: disp_width,
                    height: disp_height,
                };
                Ok(buf)
            })
            .collect::<Result<Vec<CardBuffer>, SurfaceError>>()?;
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
            property::Value::Framebuffer(buffers.first().map(|buf| buf.fb)),
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

        let surface = Self {
            card,
            buffers,
            crtc: crtc.handle(),
            show_buffer: 0,
            write_buffer: 1,
        };
        Ok(surface)
    }

    fn flip_buffer(&mut self) -> Result<(), SurfaceError> {
        let sb = &self.buffers[self.show_buffer];
        self.card
            .page_flip(self.crtc, sb.fb, PageFlipFlags::EVENT, None)?;
        let _ = self.card.receive_events();
        Ok(())
    }

    pub fn map_buffer(&mut self) -> Result<WritableSurface, SurfaceError> {
        if self.show_buffer % 2 == 0 {
            self.show_buffer = 1;
            self.write_buffer = 0;
        } else {
            self.show_buffer = 0;
            self.write_buffer = 1;
        }
        self.flip_buffer()?;
        let wb = &mut self.buffers[self.write_buffer];
        let mapping = self
            .card
            .map_dumb_buffer(&mut wb.db)
            .map_err(|err| ("Could not map dumbbuffer", err))?;
        let ws = WritableSurface {
            width: wb.width,
            height: wb.height,
            mapping,
        };
        Ok(ws)
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        for buf in &self.buffers {
            let _ = self.card.destroy_framebuffer(buf.fb);
            let _ = self.card.destroy_dumb_buffer(buf.db);
        }
    }
}

impl WritableSurface<'_> {
    pub fn width(&self) -> usize {
        self.width as usize
    }

    pub fn height(&self) -> usize {
        self.height as usize
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: (u8, u8, u8)) {
        let start = (y * self.width as usize + x) * 4;
        let target = &mut self.mapping[start..start + 4];
        let (r, g, b) = rgb;
        // Data is stored as a little-endian u32.
        target[0] = b;
        target[1] = g;
        target[2] = r;
        target[3] = 255;
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
