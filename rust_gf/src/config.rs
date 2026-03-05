use regex::{Regex, RegexBuilder};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};

#[derive(Debug, Default)]
pub struct Config {
    pub ignore_strings: HashSet<String>,
    pub target_strings: HashSet<String>,
    pub ignore_regexes: Vec<Regex>,
    pub target_regexes: Vec<Regex>,
}

impl Config {
    pub fn new() -> Self {
        let mut config = Config::default();
        config.load_configs();
        config
    }

    pub fn compile(&mut self) {
        self.ignore_regexes = self.ignore_strings
            .iter()
            .filter_map(|s| RegexBuilder::new(s).case_insensitive(true).build().ok())
            .collect();
        
        self.target_regexes = self.target_strings
            .iter()
            .filter_map(|s| RegexBuilder::new(s).case_insensitive(true).build().ok())
            .collect();
    }

    fn load_configs(&mut self) {
        let system_config = Path::new("/etc/gfconf");
        if system_config.exists() {
            self.parse_file(system_config);
        }

        if let Some(home_dir) = dirs::home_dir() {
            let user_config = home_dir.join(".gfconf");
            if user_config.exists() {
                self.parse_file(&user_config);
            }
        }
    }

    fn parse_file(&mut self, path: &Path) {
        if let Ok(file) = File::open(path) {
            let reader = io::BufReader::new(file);
            for line in reader.lines().flatten() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }

                if let Some(source_path) = line.strip_prefix("source ") {
                    let source_path = source_path.trim();
                    let path = PathBuf::from(source_path);
                    if path.exists() {
                        self.parse_file(&path);
                    }
                } else if let Some(target) = line.strip_prefix("target ") {
                    self.target_strings.insert(target.trim().to_string());
                } else if let Some(ignore) = line.strip_prefix("ignore ") {
                    self.ignore_strings.insert(ignore.trim().to_string());
                }
            }
        }
    }
}
