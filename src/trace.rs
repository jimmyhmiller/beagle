#[cfg(feature = "trace")]
use std::collections::HashSet;
#[cfg(feature = "trace")]
use std::sync::OnceLock;
#[cfg(feature = "trace")]
use std::time::Instant;

#[cfg(feature = "trace")]
static GLOBAL_TRACE_FILTER: OnceLock<TraceFilter> = OnceLock::new();
#[cfg(feature = "trace")]
static GLOBAL_TRACE_START: OnceLock<Instant> = OnceLock::new();

#[cfg(feature = "trace")]
pub struct TraceFilter {
    enabled: bool,
    categories: Option<HashSet<String>>,
}

#[cfg(feature = "trace")]
impl TraceFilter {
    pub fn from_env() -> Self {
        match std::env::var("BEAGLE_TRACE") {
            Ok(val) if val == "*" => TraceFilter {
                enabled: true,
                categories: None,
            },
            Ok(val) if !val.is_empty() => TraceFilter {
                enabled: true,
                categories: Some(val.split(',').map(|s| s.trim().to_string()).collect()),
            },
            _ => TraceFilter {
                enabled: false,
                categories: None,
            },
        }
    }

    pub fn is_enabled(&self, category: &str) -> bool {
        self.enabled
            && self
                .categories
                .as_ref()
                .map_or(true, |cats| cats.contains(category))
    }
}

#[cfg(feature = "trace")]
pub fn get_filter() -> &'static TraceFilter {
    GLOBAL_TRACE_FILTER.get_or_init(TraceFilter::from_env)
}

#[cfg(feature = "trace")]
pub fn get_start() -> &'static Instant {
    GLOBAL_TRACE_START.get_or_init(Instant::now)
}

#[cfg(feature = "trace")]
#[macro_export]
macro_rules! trace {
    ($category:expr, $($arg:tt)*) => {{
        let filter = $crate::trace::get_filter();
        if filter.is_enabled($category) {
            let elapsed = $crate::trace::get_start().elapsed();
            eprintln!(
                "[trace][{}][{:?}][{:.6}s] {}",
                $category,
                std::thread::current().id(),
                elapsed.as_secs_f64(),
                format_args!($($arg)*)
            );
        }
    }};
}

#[cfg(not(feature = "trace"))]
#[macro_export]
macro_rules! trace {
    ($category:expr, $($arg:tt)*) => {{}};
}
