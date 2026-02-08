use std::{
    mem,
    sync::atomic::{AtomicBool, Ordering},
};

pub fn listen_to_sigint() {
    unsafe {
        let mut new_action: libc::sigaction = mem::zeroed();
        new_action.sa_sigaction = handle as *const () as usize;
        new_action.sa_flags = libc::SA_SIGINFO;

        libc::sigaction(
            libc::SIGINT,
            &mut new_action as *mut libc::sigaction,
            std::ptr::null_mut(),
        );
    }
}

static RUNNING: AtomicBool = AtomicBool::new(false);

pub fn stop_requested() -> bool {
    RUNNING.load(Ordering::Relaxed)
}

extern "C" fn handle(
    _signal: libc::c_int,
    _info: *mut libc::siginfo_t,
    _context: *mut libc::c_void,
) {
    RUNNING.store(true, Ordering::Relaxed);
}
