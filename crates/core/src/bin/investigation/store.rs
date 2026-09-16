use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use bayesite_core::error::{Error, ErrorKind};
use bayesite_core::investigation::identity::{artifact_digest, Digest};
use bayesite_core::investigation::manifest::{ArtifactKind, ArtifactRef, MAX_OBJECT_BYTES};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn host_error(message: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidSettings, message)
}

pub fn objects_root(root: &Path) -> PathBuf {
    root.join("objects").join("sha256")
}

pub fn object_path(root: &Path, digest: &Digest) -> PathBuf {
    objects_root(root).join(digest.as_str())
}

pub fn reference(bytes: &[u8], kind: ArtifactKind, format: &str) -> Result<ArtifactRef, Error> {
    if bytes.len() > MAX_OBJECT_BYTES {
        return Err(host_error(format!(
            "artifact has {} bytes, exceeding the {MAX_OBJECT_BYTES}-byte limit",
            bytes.len()
        )));
    }
    Ok(ArtifactRef {
        sha256: artifact_digest(bytes),
        bytes: bytes.len(),
        kind,
        format: format.to_string(),
    })
}

pub fn insert(
    root: &Path,
    bytes: &[u8],
    kind: ArtifactKind,
    format: &str,
) -> Result<ArtifactRef, Error> {
    let reference = reference(bytes, kind, format)?;
    let directory = objects_root(root);
    fs::create_dir_all(&directory).map_err(|error| {
        host_error(format!(
            "cannot create object directory {:?}: {error}",
            directory
        ))
    })?;
    let final_path = object_path(root, &reference.sha256);
    if final_path.exists() {
        read(root, &reference)?;
        return Ok(reference);
    }

    let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = directory.join(format!(
        ".tmp-{}-{}-{}",
        std::process::id(),
        id,
        reference.sha256.as_str()
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| host_error(format!("cannot create temporary object: {error}")))?;
    if let Err(error) = file.write_all(bytes).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(host_error(format!(
            "cannot write temporary object: {error}"
        )));
    }
    drop(file);

    // A hard link is an atomic create-new publication on the trusted local
    // filesystem. The temporary name is never a valid object path. If another
    // writer won the race, verify its bytes rather than replacing it.
    match fs::hard_link(&temporary, &final_path) {
        Ok(()) => {}
        Err(_error) if final_path.exists() => {
            let _ = fs::remove_file(&temporary);
            read(root, &reference)?;
            return Ok(reference);
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(host_error(format!(
                "cannot publish object {} without replacement: {error}",
                reference.sha256.prefixed()
            )));
        }
    }
    let _ = fs::remove_file(&temporary);
    Ok(reference)
}

pub fn read_digest(root: &Path, digest: &Digest) -> Result<Vec<u8>, Error> {
    let path = object_path(root, digest);
    let metadata = fs::metadata(&path).map_err(|error| {
        host_error(format!(
            "cannot inspect object {} at {:?}: {error}",
            digest.prefixed(),
            path
        ))
    })?;
    if metadata.len() > MAX_OBJECT_BYTES as u64 {
        return Err(host_error(format!(
            "object {} exceeds the {MAX_OBJECT_BYTES}-byte limit",
            digest.prefixed()
        )));
    }
    let bytes = fs::read(&path).map_err(|error| {
        host_error(format!(
            "cannot read object {} at {:?}: {error}",
            digest.prefixed(),
            path
        ))
    })?;
    let actual = artifact_digest(&bytes);
    if actual != *digest {
        return Err(host_error(format!(
            "object {} is corrupt (hashes to {}); restore the exact object",
            digest.prefixed(),
            actual.prefixed()
        )));
    }
    Ok(bytes)
}

pub fn read(root: &Path, reference: &ArtifactRef) -> Result<Vec<u8>, Error> {
    let path = object_path(root, &reference.sha256);
    let bytes = read_digest(root, &reference.sha256).map_err(|error| {
        host_error(format!(
            "cannot read object {} at {:?}: {error}",
            reference.sha256.prefixed(),
            path
        ))
    })?;
    if bytes.len() != reference.bytes {
        return Err(host_error(format!(
            "object {} has {} bytes but reference requires {}; restore the exact object",
            reference.sha256.prefixed(),
            bytes.len(),
            reference.bytes
        )));
    }
    Ok(bytes)
}

pub fn import_digest(
    source_root: &Path,
    destination_root: &Path,
    digest: &Digest,
) -> Result<(), Error> {
    let bytes = read_digest(source_root, digest)?;
    let inserted = insert(
        destination_root,
        &bytes,
        ArtifactKind::EngineBinary,
        "opaque-citation",
    )?;
    if inserted.sha256 != *digest {
        return Err(host_error("imported citation digest changed unexpectedly"));
    }
    Ok(())
}

pub fn import_reference(
    source_root: &Path,
    destination_root: &Path,
    reference: &ArtifactRef,
) -> Result<(), Error> {
    let bytes = read(source_root, reference)?;
    let inserted = insert(destination_root, &bytes, reference.kind, &reference.format)?;
    if inserted != *reference {
        return Err(host_error("imported object metadata changed unexpectedly"));
    }
    Ok(())
}

pub fn import_all(source_root: &Path, destination_root: &Path) -> Result<usize, Error> {
    let source = objects_root(source_root);
    let mut count = 0usize;
    for entry in fs::read_dir(&source).map_err(|error| {
        host_error(format!(
            "cannot list object directory {:?}: {error}",
            source
        ))
    })? {
        let entry = entry.map_err(|error| host_error(format!("cannot list object: {error}")))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| host_error("object filename must be UTF-8"))?;
        let digest = match Digest::parse(&name, "object filename") {
            Ok(digest) => digest,
            Err(_) if name.starts_with(".tmp-") => continue,
            Err(error) => return Err(error),
        };
        let bytes = fs::read(entry.path())
            .map_err(|error| host_error(format!("cannot read source object {name}: {error}")))?;
        if artifact_digest(&bytes) != digest {
            return Err(host_error(format!(
                "source object {} is corrupt; verification is required before import",
                digest.prefixed()
            )));
        }
        // Kind/format are not encoded in the store path. Raw import preserves
        // exact bytes; typed verification follows through manifest references.
        let destination = objects_root(destination_root).join(digest.as_str());
        fs::create_dir_all(objects_root(destination_root)).map_err(|error| {
            host_error(format!(
                "cannot create destination object directory: {error}"
            ))
        })?;
        if destination.exists() {
            let existing = fs::read(&destination)
                .map_err(|error| host_error(format!("cannot read destination object: {error}")))?;
            if existing != bytes {
                return Err(host_error(format!(
                    "destination object {} exists with different bytes; refusing to overwrite",
                    digest.prefixed()
                )));
            }
        } else {
            let synthetic = insert(
                destination_root,
                &bytes,
                ArtifactKind::EngineBinary,
                "opaque-import",
            )?;
            debug_assert_eq!(synthetic.sha256, digest);
        }
        count += 1;
    }
    Ok(count)
}
