use std::{
    array, error, fmt,
    fs::{File, OpenOptions},
    io, mem,
    sync::mpsc,
    thread,
};

use rand::prelude::*;

use rustix::ioctl;

pub struct Player {
    join_handle: Option<thread::JoinHandle<Result<(), SoundError>>>,
    shutdown_chan: mpsc::Sender<()>,
}

const SAMPLING_FREQUENCY: u32 = 48000;
const SAMPLE_SECONDS: u32 = 2;
const CHANNELS: u32 = 2;
const BIQUAD_LOWPASS_CENTER_FREQUENCY: f32 = 150.0;
const BIQUAD_LOWPASS_Q_FACTOR: f32 = 3.0;
const BIQUAD_MULTIPLICATION: f32 = 1.0;
const NOISE_SAMPLES: usize = (SAMPLING_FREQUENCY * CHANNELS * SAMPLE_SECONDS) as usize;

impl Player {
    pub fn new(filename: &str) -> Result<Player, io::Error> {
        let device = Device::create(filename)?;

        unsafe {
            device.set_params()?;
        }

        let (shutdown_sender, shutdown_receiver) = mpsc::channel();
        let join_handle = thread::spawn(move || unsafe {
            let mut device = device;
            device.play_loop(shutdown_receiver)
        });
        Ok(Player {
            join_handle: Some(join_handle),
            shutdown_chan: shutdown_sender,
        })
    }
}

struct Device {
    file: File,
}

impl Device {
    fn create(filename: &str) -> Result<Device, io::Error> {
        // Based on https://github.com/PHJArea217/raw-alsa-player/blob/master/raw-alsa-player.c
        // Also, check https://github.com/psqli/pcm.sh for another example
        let file = OpenOptions::new().read(true).write(true).open(filename)?;

        Ok(Device { file })
    }

    unsafe fn set_params(&self) -> Result<(), io::Error> {
        let mut params = alsa::Params::new();
        params.rmask = 0;
        params.cmask = 0;
        params.info = 0;

        // Setting an unsupported format might cause device to refuse ioctl.
        let format = 2; // SNDRV_PCM_FORMAT_S16_LE
        let sample_length = std::mem::size_of::<i16>() as u32;
        let frame_len = sample_length * CHANNELS;
        let access = 3; // SNDRV_PCM_ACCESS_RW_INTERLEAVED

        params.set_imask(
            alsa::SNDRV_PCM_HW_PARAM_SAMPLE_BITS,
            i16::BITS,
            false,
            false,
        );
        params.set_imask(
            alsa::SNDRV_PCM_HW_PARAM_FRAME_BITS,
            frame_len * 8,
            false,
            false,
        );
        params.set_imask(
            alsa::SNDRV_PCM_HW_PARAM_RATE,
            SAMPLING_FREQUENCY,
            false,
            false,
        );
        params.set_imask(alsa::SNDRV_PCM_HW_PARAM_CHANNELS, CHANNELS, false, false);
        params.set_imask(alsa::SNDRV_PCM_HW_PARAM_SUBFORMAT, 0, false, false);
        params.set_imask(alsa::SNDRV_PCM_HW_PARAM_ACCESS, access, false, false);
        params.set_imask(alsa::SNDRV_PCM_HW_PARAM_FORMAT, format, false, false);

        ioctl::ioctl(
            &self.file,
            ioctl::Updater::<alsa::IoCtlHwParams, alsa::Params>::new(&mut params),
        )?;

        ioctl::ioctl(&self.file, ioctl::NoArg::<alsa::IoCtlPrepare>::new())?;
        Ok(())
    }

    unsafe fn play_loop(&mut self, shutdown_chan: mpsc::Receiver<()>) -> Result<(), SoundError> {
        // This loop will non-blockingly check if a new buffer is available and switch to using it.
        // The used buffer will be send to the generator, so that it can fill it with fresh data;
        // otherwise, keep playing the current buffer in a loop.
        let (new_buffer_requester, empty_buffer_receiver) = mpsc::channel();
        let (full_buffer_sender, full_buffer_receiver) = mpsc::channel();
        let generator_join_handle = thread::spawn(move || {
            Device::generator_loop(empty_buffer_receiver, full_buffer_sender)
        });
        // Prefill the buffer queue with a few empty buffers.
        (0..3).for_each(|_| {
            let _ = new_buffer_requester.send(vec![0i16; NOISE_SAMPLES]);
        });
        // Wait until first buffer is available.
        let mut noise_samples = full_buffer_receiver.recv()?;
        loop {
            match shutdown_chan.try_recv() {
                Ok(_) => {
                    break;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    break;
                }
                _ => {}
            }
            match full_buffer_receiver.try_recv() {
                Ok(new_buffer) => {
                    let mut new_buffer = new_buffer;
                    mem::swap(&mut new_buffer, &mut noise_samples);
                    let _ = new_buffer_requester.send(new_buffer);
                }
                Err(mpsc::TryRecvError::Empty) => {}
                Err(mpsc::TryRecvError::Disconnected) => {
                    break;
                }
            }

            let submit = alsa::SubmitBuffer::new(noise_samples.as_slice());
            match ioctl::ioctl(
                &self.file,
                ioctl::Setter::<alsa::IoCtlWriteiFrames, _>::new(submit),
            ) {
                Ok(()) => {}
                Err(rustix::io::Errno::PIPE) => {
                    // Buffer underrun occurred, restart device.
                    ioctl::ioctl(&self.file, ioctl::NoArg::<alsa::IoCtlPrepare>::new())?;
                }
                Err(rustix::io::Errno::INTR) => {
                    // If interrupted, nothing to do.
                }
                Err(err) => {
                    return Err(err.into());
                }
            }
        }
        drop(new_buffer_requester);
        drop(full_buffer_receiver);
        let _ = generator_join_handle.join();

        Ok(())
    }

    fn generator_loop(
        empty_buffers_chan: mpsc::Receiver<Vec<i16>>,
        full_buffers_chan: mpsc::Sender<Vec<i16>>,
    ) -> Result<(), mpsc::RecvError> {
        // Biquads are stateful and playing the same fragment in a loop
        // will cause popping or crackling noise.
        let mut rng = rand::thread_rng();
        let mut biquads: [Biquad; CHANNELS as usize] = array::from_fn(|_| {
            Biquad::new(BIQUAD_LOWPASS_CENTER_FREQUENCY, BIQUAD_LOWPASS_Q_FACTOR)
        });
        let biquad_multiplication: f32 = BIQUAD_MULTIPLICATION * (i16::MAX as f32);

        loop {
            let mut noise_dest = empty_buffers_chan.recv()?;
            noise_dest.iter_mut().enumerate().for_each(|(i, dest)| {
                let biquad = &mut biquads[i % (CHANNELS as usize)];
                let noise_sample = rng.gen_range(-1.0..1.0);
                let out = biquad.process(noise_sample) * biquad_multiplication;
                *dest = out.clamp(i16::MIN as f32, i16::MAX as f32) as i16;
            });
            let _ = full_buffers_chan.send(noise_dest);
        }
    }
}

struct Biquad {
    y1: f32,
    y2: f32,
    x1: f32,
    x2: f32,

    a1: f32,
    a2: f32,
    b0: f32,
    b1: f32,
    b2: f32,
}

impl Biquad {
    fn new(lowpass_frequency: f32, q: f32) -> Biquad {
        let omega = 2.0 * std::f32::consts::PI * (lowpass_frequency / SAMPLING_FREQUENCY as f32);
        let omega_s = omega.sin();
        let omega_c = omega.cos();
        let alpha = omega_s / (2.0 * q);

        let b0 = (1.0 - omega_c) * 0.5;
        let b1 = 1.0 - omega_c;
        let b2 = (1.0 - omega_c) * 0.5;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * omega_c;
        let a2 = 1.0 - alpha;

        Biquad {
            y1: 0.0,
            y2: 0.0,
            x1: 0.0,
            x2: 0.0,
            a1: a1 / a0,
            a2: a2 / a0,
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
        }
    }

    fn process(&mut self, val: f32) -> f32 {
        // Biquad Direct Form 1
        let out = self.b0 * val + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = val;
        self.y2 = self.y1;
        self.y1 = out;

        out
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        if self.shutdown_chan.send(()).is_err() {
            return;
        }
        if let Some(join_handle) = self.join_handle.take() {
            let _ = join_handle.join();
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            let _ = ioctl::ioctl(&self.file, ioctl::NoArg::<alsa::IoCtlDrain>::new());
            let _ = ioctl::ioctl(&self.file, ioctl::NoArg::<alsa::IoCtlDrop>::new());
        }
    }
}

mod alsa {
    use libc::c_void;
    use rustix::ioctl;
    use std::array;

    use super::CHANNELS;

    pub const SNDRV_PCM_HW_PARAM_ACCESS: usize = 0;
    pub const SNDRV_PCM_HW_PARAM_FORMAT: usize = 1;
    pub const SNDRV_PCM_HW_PARAM_SUBFORMAT: usize = 2;
    pub const SNDRV_PCM_HW_PARAM_FIRST_MASK: usize = SNDRV_PCM_HW_PARAM_ACCESS;
    pub const SNDRV_PCM_HW_PARAM_LAST_MASK: usize = SNDRV_PCM_HW_PARAM_SUBFORMAT;
    pub const SNDRV_PCM_HW_PARAM_SAMPLE_BITS: usize = 8;
    pub const SNDRV_PCM_HW_PARAM_FRAME_BITS: usize = 9;
    pub const SNDRV_PCM_HW_PARAM_CHANNELS: usize = 10;
    pub const SNDRV_PCM_HW_PARAM_RATE: usize = 11;
    pub const SNDRV_PCM_HW_PARAM_TICK_TIME: usize = 19;

    pub const SNDRV_PCM_HW_PARAM_FIRST_INTERVAL: usize = SNDRV_PCM_HW_PARAM_SAMPLE_BITS;
    pub const SNDRV_PCM_HW_PARAM_LAST_INTERVAL: usize = SNDRV_PCM_HW_PARAM_TICK_TIME;

    type SoundMask = [u32; (256 + 31) / 32];

    pub struct SoundInterval {
        pub min: u32,
        pub max: u32,
        pub flags: u32, // snd_interval uses C bitfields here
    }

    #[repr(C)]
    pub struct Params {
        pub flags: u32,
        pub masks: [SoundMask; SNDRV_PCM_HW_PARAM_LAST_MASK - SNDRV_PCM_HW_PARAM_FIRST_MASK + 1],
        _masks_reserved: [SoundMask; 5],
        pub intervals: [SoundInterval;
            SNDRV_PCM_HW_PARAM_LAST_INTERVAL - SNDRV_PCM_HW_PARAM_FIRST_INTERVAL + 1],
        _intervals_reserved: [SoundInterval; 9],
        pub rmask: u32,
        pub cmask: u32,
        pub info: u32,
        pub msbits: u32,
        pub rate_num: u32,
        pub rate_den: u32,
        pub fifo_size: u64,
        _reserved: [u8; 64],
    }

    impl Params {
        pub fn new() -> Params {
            let new_soundmask = |_| array::from_fn(|i| if i == 0 { !0u32 } else { 0 });
            let new_interval = |_| SoundInterval {
                min: 0,
                max: !0u32,
                flags: 0,
            };

            Params {
                flags: 0,
                masks: array::from_fn(new_soundmask),
                _masks_reserved: array::from_fn(new_soundmask),
                intervals: array::from_fn(new_interval),
                _intervals_reserved: array::from_fn(new_interval),
                rmask: 0,
                cmask: 0,
                info: 0,
                msbits: 0,
                rate_num: 0,
                rate_den: 0,
                fifo_size: 0,
                _reserved: [0; 64],
            }
        }

        pub fn set_imask(&mut self, param: usize, value: u32, min: bool, max: bool) {
            if param >= SNDRV_PCM_HW_PARAM_FIRST_INTERVAL {
                let interval = &mut self.intervals[param - SNDRV_PCM_HW_PARAM_FIRST_INTERVAL];
                interval.min = value;
                interval.max = value;
                let openmin = if min { 1 << 0 } else { 0 };
                let openmax = if max { 1 << 1 } else { 0 };
                let integer = if !(min || max) { 1 << 2 } else { 0 };
                interval.flags = openmin | openmax | integer;
            } else {
                let mask = &mut self.masks[param - SNDRV_PCM_HW_PARAM_FIRST_MASK];
                mask.iter_mut().for_each(|val| *val = 0);
                // FD_SET just sets the i-th bit in an array.
                let index = value / 32;
                let offset = value % 32;
                let dst = &mut mask[index as usize];
                // TODO: provide a better option to toggle flags on/off?
                (*dst) |= 1u32 << offset;
            }
            self.rmask |= 1 << param;
        }
    }

    #[repr(C)]
    pub struct SubmitBuffer {
        result: i64,
        buffer: *const c_void,
        num_frames: u64,
    }

    impl SubmitBuffer {
        pub fn new(data: &[i16]) -> SubmitBuffer {
            SubmitBuffer {
                result: 0,
                buffer: data.as_ptr() as *const c_void,
                num_frames: data.len() as u64 / CHANNELS as u64,
            }
        }
    }

    pub type IoCtlHwParams = ioctl::ReadWriteOpcode<b'A', 0x11, Params>;
    pub type IoCtlPrepare = ioctl::NoneOpcode<b'A', 0x40, ()>;
    pub type IoCtlDrop = ioctl::NoneOpcode<b'A', 0x43, ()>;
    pub type IoCtlDrain = ioctl::NoneOpcode<b'A', 0x44, ()>;
    pub type IoCtlWriteiFrames<'a> = ioctl::WriteOpcode<b'A', 0x50, SubmitBuffer>;
}

#[derive(Debug)]
pub enum SoundError {
    Recv(mpsc::RecvError),
    Io(io::Error),
    Ioctl(rustix::io::Errno),
}

impl fmt::Display for SoundError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            SoundError::Recv(ref e) => {
                write!(f, "Receive error: {}", e)
            }
            SoundError::Io(ref e) => {
                write!(f, "IO error: {}", e)
            }
            SoundError::Ioctl(ref e) => {
                write!(f, "IOCTL error: {}", e)
            }
        }
    }
}

impl error::Error for SoundError {
    fn cause(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            SoundError::Recv(ref e) => Some(e),
            SoundError::Io(ref e) => Some(e),
            SoundError::Ioctl(ref e) => Some(e),
        }
    }
}

impl From<mpsc::RecvError> for SoundError {
    fn from(err: mpsc::RecvError) -> SoundError {
        SoundError::Recv(err)
    }
}

impl From<io::Error> for SoundError {
    fn from(err: io::Error) -> SoundError {
        SoundError::Io(err)
    }
}

impl From<rustix::io::Errno> for SoundError {
    fn from(err: rustix::io::Errno) -> SoundError {
        SoundError::Ioctl(err)
    }
}
