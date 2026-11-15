//! Minimal command line parsing: positional arguments and
//! `--name value` flags.
use std::collections::HashMap;
use std::str::FromStr;

pub struct Args {
    positional: Vec<String>,
    flags: HashMap<String, String>,
}

impl Args {
    pub fn parse(mut it: impl Iterator<Item = String>) -> Args {
        let mut args = Args { positional: vec![], flags: HashMap::new() };
        while let Some(s) = it.next() {
            match s.strip_prefix("--") {
                Some(name) => {
                    let value = it.next().unwrap_or_else(|| panic!("--{name} needs a value"));
                    args.flags.insert(name.to_string(), value);
                }
                None => args.positional.push(s),
            }
        }
        args
    }

    pub fn positional(&self, i: usize) -> Option<&str> {
        self.positional.get(i).map(String::as_str)
    }

    /// Takes the flag's value, or `default` when absent.
    pub fn flag<T: FromStr>(&mut self, name: &str, default: T) -> T {
        self.flags.remove(name).map_or(default, |v| {
            v.parse().unwrap_or_else(|_| panic!("bad value for --{name}: {v}"))
        })
    }

    /// Panics on flags that were never taken, which catches typos.
    pub fn finish(self) {
        let unknown: Vec<_> = self.flags.keys().collect();
        assert!(unknown.is_empty(), "unknown flags: {unknown:?}");
    }
}
