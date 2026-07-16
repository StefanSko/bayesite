# Releasing Bayesite

Bayesite releases one versioned Rust CLI as four platform archives plus
sha256 sidecars. A `vX.Y.Z` tag triggers `.github/workflows/release.yml`.

## Changelog discipline

- Every user-visible PR updates the `Unreleased` section of `CHANGELOG.md`, or
  states in its PR description why no entry is needed.
- Engineering logs such as `implementation-notes.md` are not release notes.
- Release preparation moves all `Unreleased` entries into one dated version
  section and restores an empty `Unreleased` section.
- Historical release sections are not rewritten except to correct factual
  errors.

## Preconditions

1. The Bayeswire vendor pin names the intended upstream commit and
   `python3 scripts/check_validation_ladder.py` passes.
2. The release section in `CHANGELOG.md` describes every user-visible change
   since the previous tag.
3. The crate version and `Cargo.lock` agree with the intended tag.
4. Main CI is green.
5. The release workflow's RustSec gate passes against the current advisory
   database with no vulnerability or warning-class advisory; retain its logged
   database commit and date with the release run.

## Cut a release

1. On a release-preparation branch, update `crates/core/Cargo.toml`, refresh
   `Cargo.lock` with Cargo, and update versioned install/capabilities examples
   in `README.md` and `docs/capabilities-v0.md`. Release-tooling tests enforce
   that these versions agree.
2. Finalize the changelog section, run the full validation ladder, and merge a
   reviewed release-preparation PR.
3. Tag the resulting `main` commit:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin main vX.Y.Z
   ```

4. Wait for the release workflow to validate, build, smoke-test, and publish
   all four archives and checksum sidecars.
5. Replace the workflow's provisional GitHub Release notes with the matching
   `CHANGELOG.md` section.
6. Download at least the current platform archive, verify its sidecar, and run
   `bayesite capabilities`.
7. Only after the assets exist, update Bayescycle with
   `packages/bayescycle/scripts/bump_engine_release.py --tag vX.Y.Z`.
