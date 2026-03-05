use colored::*;
use content_inspector::{inspect, ContentType};
use regex::Regex;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, BufRead, Read};
use std::path::Path;
use walkdir::WalkDir;

use crate::config::Config;

const BINARY_EXTENSIONS: &[&str] = &[
    "exe", "dll", "so", "dylib", "bin", "obj", "o", "a", "lib", "jar", "war", "ear", "class",
    "pyc", "pyo", "pyd", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "ico", "svg", "webp", "mp4",
    "avi", "mov", "wmv", "flv", "mp3", "wav", "flac", "ogg", "m4a", "zip", "rar", "7z", "tar",
    "gz", "bz2", "xz", "lzma",
];

pub struct SearchOptions {
    pub pattern: Regex,
    pub context: usize,
    pub max_line: Option<usize>,
    pub debug: bool,
}

pub fn recursive_search(config: &Config, options: &SearchOptions) {
    #[cfg(unix)]
    let stdout_stat = {
        use std::os::unix::fs::MetadataExt;
        std::fs::metadata("/dev/stdout").ok().map(|m| (m.dev(), m.ino()))
    };
    #[cfg(not(unix))]
    let stdout_stat: Option<(u64, u64)> = None;

    for entry in WalkDir::new(".").follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                if options.debug {
                    eprintln!("Error accessing entry: {}", e);
                }
                continue;
            }
        };

        let path = entry.path();

        if should_skip(path, config, options.debug) {
            continue;
        }

        if path.is_file() {
            #[cfg(unix)]
            {
                if let Some((out_dev, out_ino)) = stdout_stat {
                    if let Ok(meta) = std::fs::metadata(path) {
                        use std::os::unix::fs::MetadataExt;
                        if meta.dev() == out_dev && meta.ino() == out_ino {
                            if options.debug {
                                eprintln!("Skipping output file: {}", path.display());
                            }
                            continue;
                        }
                    }
                }
            }

            if let Err(e) = search_file(path, options) {
                if options.debug {
                    eprintln!("Error reading file {}: {}", path.display(), e);
                }
            }
        }
    }
}

fn should_skip(path: &Path, config: &Config, debug: bool) -> bool {
    let path_str = path.to_string_lossy();

    // 1. Check ignore regexes (pre-compiled)
    for re in &config.ignore_regexes {
        if re.is_match(&path_str) {
            if debug { println!("Skipping {} (ignored by pattern)", path.display()); }
            return true;
        }
    }

    // 2. Check target regexes (pre-compiled)
    if path.is_file() && !config.target_regexes.is_empty() {
        let mut matched_target = false;
        for re in &config.target_regexes {
            if re.is_match(&path_str) {
                matched_target = true;
                break;
            }
        }
        if !matched_target {
            if debug { println!("Skipping {} (no target match)", path.display()); }
            return true;
        }
    }
    
    // 3. Check binary extensions
    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            if BINARY_EXTENSIONS.contains(&ext_str.to_lowercase().as_str()) {
                if debug { println!("Skipping {} (binary extension)", path.display()); }
                return true;
            }
        }
    }

    false
}

fn is_binary_file(path: &Path) -> bool {
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return true,
    };
    
    let mut buffer = [0; 1024];
    let n = match file.read(&mut buffer) {
        Ok(n) => n,
        Err(_) => return true,
    };

    inspect(&buffer[..n]) == ContentType::BINARY
}


fn search_file(path: &Path, options: &SearchOptions) -> io::Result<()> {
    if is_binary_file(path) {
        return Ok(());
    }

    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut line_num = 0;
    let mut context_queue: VecDeque<(usize, String)> = VecDeque::new();
    let mut lines_since_match = options.context + 1;
    let mut printed_filename = false;

    for line_res in reader.lines() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => return Ok(()),
        };
        line_num += 1;

        if let Some(max) = options.max_line {
            if line_num > max {
                break;
            }
        }

        if line.len() > 1_000_000 {
            return Ok(());
        }

        let is_match = options.pattern.is_match(&line);

        if is_match {
            if !printed_filename {
                println!("{}", path.display().to_string().green());
                printed_filename = true;
            }

            while let Some((ctx_num, ctx_line)) = context_queue.pop_front() {
                 println!("[{}]\t{}", ctx_num, ctx_line);
            }

            print!("[{}]\t", line_num);
            let mut last_match_end = 0;
            for mat in options.pattern.find_iter(&line) {
                print!("{}", &line[last_match_end..mat.start()]);
                print!("{}", &line[mat.start()..mat.end()].red().bold());
                last_match_end = mat.end();
            }
            println!("{}", &line[last_match_end..]);

            lines_since_match = 0;
        } else {
            if lines_since_match < options.context {
                 println!("[{}]\t{}", line_num, line);
                 lines_since_match += 1;
            } else if options.context > 0 {
                context_queue.push_back((line_num, line));
                if context_queue.len() > options.context {
                    context_queue.pop_front();
                }
            }
        }
    }
    
    if printed_filename {
        println!();
    }

    Ok(())
}
