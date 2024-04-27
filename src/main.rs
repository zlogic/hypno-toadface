use drm::buffer::DrmFourcc;
use drm::control;
use drm::control::atomic;
use drm::control::connector;
use drm::control::crtc;
use drm::control::property;
use drm::control::AtomicCommitFlags;
use drm::control::Device as ControlDevice;
use drm::control::PageFlipFlags;
use drm::Device;

use std::os::unix::io::AsFd;
use std::os::unix::io::BorrowedFd;

pub struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl Device for Card {}
impl ControlDevice for Card {}

impl Card {
    pub fn open(path: &str) -> Self {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        options.write(true);
        Card(options.open(path).unwrap())
    }
}

fn main() {
    let card = Card::open("/dev/dri/card1");
    println!("{:#?}", card.get_driver().unwrap());

    card.set_client_capability(drm::ClientCapability::UniversalPlanes, true)
        .expect("Unable to request UniversalPlanes capability");
    card.set_client_capability(drm::ClientCapability::Atomic, true)
        .expect("Unable to request Atomic capability");

    // Load the information.
    let res = card
        .resource_handles()
        .expect("Could not load normal resource ids.");
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
        .expect("No connected connectors");

    // Get the first (usually best) mode
    let &mode = con.modes().first().expect("No modes found on connector");

    let (disp_width, disp_height) = mode.size();

    // Find a crtc and FB
    let crtc = crtcinfo.first().expect("No crtcs found");

    // Select the pixel format
    let fmt = DrmFourcc::Xrgb8888;

    // Create a DB
    // If buffer resolution is above display resolution, a ENOSPC (not enough GPU memory) error may
    // occur
    let mut db1 = card
        .create_dumb_buffer((disp_width.into(), disp_height.into()), fmt, 32)
        .expect("Could not create dumb buffer");
    let mut db2 = card
        .create_dumb_buffer((disp_width.into(), disp_height.into()), fmt, 32)
        .expect("Could not create dumb buffer");

    // Map it and grey it out.
    {
        let mut map = card
            .map_dumb_buffer(&mut db1)
            .expect("Could not map dumbbuffer");
        for b in map.as_mut() {
            *b = 128;
        }
    }

    // Create an FB:
    let fb1 = card
        .add_framebuffer(&db1, 24, 32)
        .expect("Could not create FB");
    let fb2 = card
        .add_framebuffer(&db2, 24, 32)
        .expect("Could not create FB");

    let planes = card.plane_handles().expect("Could not list planes");
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

    println!("{:#?}", mode);
    println!("{:#?}", fb1);
    println!("{:#?}", fb2);
    println!("{:#?}", db1);
    println!("{:#?}", db2);
    println!("{:#?}", plane);

    let con_props = card
        .get_properties(con.handle())
        .expect("Could not get props of connector")
        .as_hashmap(&card)
        .expect("Could not get a prop from connector");
    let crtc_props = card
        .get_properties(crtc.handle())
        .expect("Could not get props of crtc")
        .as_hashmap(&card)
        .expect("Could not get a prop from crtc");
    let plane_props = card
        .get_properties(plane)
        .expect("Could not get props of plane")
        .as_hashmap(&card)
        .expect("Could not get a prop from plane");

    let mut atomic_req = atomic::AtomicModeReq::new();
    atomic_req.add_property(
        con.handle(),
        con_props["CRTC_ID"].handle(),
        property::Value::CRTC(Some(crtc.handle())),
    );
    let blob = card
        .create_property_blob(&mode)
        .expect("Failed to create blob");
    atomic_req.add_property(crtc.handle(), crtc_props["MODE_ID"].handle(), blob);
    atomic_req.add_property(
        crtc.handle(),
        crtc_props["ACTIVE"].handle(),
        property::Value::Boolean(true),
    );
    atomic_req.add_property(
        plane,
        plane_props["FB_ID"].handle(),
        property::Value::Framebuffer(Some(fb1)),
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
        .expect("Failed to set mode");

    let mut color = 0u32;
    for i in 0..600 {
        let mut map = if i % 2 == 0 {
            card.map_dumb_buffer(&mut db1)
                .expect("Could not map dumbbuffer")
        } else {
            card.map_dumb_buffer(&mut db2)
                .expect("Could not map dumbbuffer")
        };
        color = (i % 2) * 3000 * 2000;
        //if i < 2 {
        for p in map.chunks_exact_mut(4) {
            p[3] = 255;
            p[2] = ((color >> 16) & 0xff) as u8;
            p[1] = ((color >> 8) & 0xff) as u8;
            p[0] = ((color >> 0) & 0xff) as u8;
            if i % 2 == 0 {
                color += 1;
            }
        }
        //}
        //let one_second = ::std::time::Duration::from_millis(1);
        //::std::thread::sleep(one_second);

        card.wait_vblank(
            drm::VblankWaitTarget::Relative(0),
            drm::VblankWaitFlags::NEXT_ON_MISS,
            0,
            0,
        );
        let mut atomic_req = atomic::AtomicModeReq::new();
        let fb = if i % 2 == 0 { fb1 } else { fb2 };

        /*
        atomic_req.add_property(
            plane,
            plane_props["FB_ID"].handle(),
            property::Value::Framebuffer(Some(fb)),
        );
        card.atomic_commit(AtomicCommitFlags::ALLOW_MODESET, atomic_req)
            .expect("Failed to set mode");
        */
        card.page_flip(crtc.handle(), fb, PageFlipFlags::empty(), None);
    }
    //let five_seconds = ::std::time::Duration::from_millis(5000);
    //::std::thread::sleep(five_seconds);

    card.destroy_framebuffer(fb1).unwrap();
    card.destroy_framebuffer(fb2).unwrap();
    card.destroy_dumb_buffer(db1).unwrap();
    card.destroy_dumb_buffer(db2).unwrap();

    let resources = card.resource_handles().unwrap();
    let plane_res = card.plane_handles().unwrap();

    for &handle in resources.framebuffers() {
        let info = card.get_framebuffer(handle).unwrap();
        println!("Framebuffer: {:?}", handle);
        println!("\tSize: {:?}", info.size());
        println!("\tPitch: {:?}", info.pitch());
        println!("\tBPP: {:?}", info.bpp());
        println!("\tDepth: {:?}", info.depth());
    }
}
