use super::*;

#[allow(unused)]
#[unsafe(no_mangle)]
#[inline(never)]
/// # Safety
///
/// This does nothing
pub unsafe extern "C" fn debugger_info(buffer: *const u8, length: usize) {
    // Hack to make sure this isn't inlined
    black_box(buffer);
    black_box(length);
}

#[macro_export]
macro_rules! debug_only {
    ($($code:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $($code)*
        }
    };
}

#[macro_export]
macro_rules! debug_flag_only {
    ($($code:tt)*) => {
        {
            let runtime = $crate::get_runtime().get();
            if runtime.get_command_line_args().debug {
                $($code)*
            }
        }
    };
}

pub fn debugger(_message: Message) {
    debug_only! {
        let serialized_message : Vec<u8>;
        #[cfg(feature="json")] {
        use nanoserde::SerJson;
            let serialized : String = SerJson::serialize_json(&_message);
            serialized_message = serialized.into_bytes();
        }
        #[cfg(not(feature="json"))] {
            use crate::Serialize;
            serialized_message = _message.to_binary();
        }
        let ptr = serialized_message.as_ptr();
        let length = serialized_message.len();
        unsafe {
            debugger_info(ptr, length);
        }
        drop(serialized_message);
    }
}
