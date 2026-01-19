# CHANGELOG

<!-- version list -->

## v3.0.5 (2026-01-19)

### Bug Fixes

- Datasets.py, verify datasets, switch to tempfile pattern, rerun readme
  ([`f5d6651`](https://github.com/sradc/SmallPebble/commit/f5d6651e0f0d6aeffca12284aeadf2ce336ca0f4))

- Type hint of conv2d strides and change default to immutable
  ([`d21e41f`](https://github.com/sradc/SmallPebble/commit/d21e41f0bb3644f6a2003fa53a677df28194531f))

### Continuous Integration

- Chore use `packages-dir` instead of `packages_dir` as warned
  ([`8b82140`](https://github.com/sradc/SmallPebble/commit/8b821401744bc18dddb886ef238fbf9c08929a3a))

### Documentation

- Update links in nb to use 'main'
  ([`e166224`](https://github.com/sradc/SmallPebble/commit/e166224a19527fd57c71b3545ecefc247f2e1245))


## v3.0.4 (2026-01-18)

### Bug Fixes

- Again, test ci/cd pipeline configuration
  ([`846c800`](https://github.com/sradc/SmallPebble/commit/846c80090e79fd4357ee4ef485beb0a0f56239ed))


## v3.0.3 (2026-01-18)

### Bug Fixes

- Test ci/cd pipeline configuration
  ([`d1fd1ea`](https://github.com/sradc/SmallPebble/commit/d1fd1ea30b849335f8e0ad264aabdbb2f160a417))

### Continuous Integration

- Add condition to release to only run on main, and a safety check to avoid running on semrelease
  commits
  ([`e49a23b`](https://github.com/sradc/SmallPebble/commit/e49a23bcc8ccc65fe356f643d034b3e632d5a2a6))

- Simplify into one file
  ([`ed0f69f`](https://github.com/sradc/SmallPebble/commit/ed0f69fc9889a985c495db9888bef84c36caea43))

- Update release.yml to only run after tests
  ([`c417751`](https://github.com/sradc/SmallPebble/commit/c417751c39c3620d2f23b77c704f6b2d8a265a83))

### Documentation

- Update readme notes on download location
  ([`371e782`](https://github.com/sradc/SmallPebble/commit/371e78282bf79211ea81cdad65468136118d25fa))


## v3.0.2 (2026-01-18)

### Bug Fixes

- Datasets now fetches from smallpebble repo release, simpler / more reliable / hopefully works in
  colab unlike the prev code
  ([`2e7004b`](https://github.com/sradc/SmallPebble/commit/2e7004b4a610ec03aaabd447f45b0e73ac248667))

### Chores

- Delete datasets_to_npz.ipynb, ds uploaded, nb no longer needed
  ([`7c9bab4`](https://github.com/sradc/SmallPebble/commit/7c9bab4fddf69790f9e6d4ee7174540d568b4605))

- Nb to convert datasets to npz files to add to release
  ([`a1b6bee`](https://github.com/sradc/SmallPebble/commit/a1b6bee674dd9d604788263c18baa250708e97cd))

### Continuous Integration

- Another fix for numpy/tensorflow dep conflict
  ([`322bdb3`](https://github.com/sradc/SmallPebble/commit/322bdb3b63de0b45bdc454d348666186fd4830fb))

- Fix tensorflow/numpy deps, rename to test
  ([`914038f`](https://github.com/sradc/SmallPebble/commit/914038fc12cc93e3da62d350866b351e232733bc))

### Documentation

- Add pypi badge and instructions to readme
  ([`838d448`](https://github.com/sradc/SmallPebble/commit/838d448fe3c624fd7d95238a36691a098ef283dd))

- Collab link and readme tweaks
  ([`65016f7`](https://github.com/sradc/SmallPebble/commit/65016f7cb33d1433b17d6a53226d382cfebbc3c4))

- Improve readme link at top of readme
  ([`b4639b1`](https://github.com/sradc/SmallPebble/commit/b4639b1472c4580615c8751b2c569984875e7596))

- Move readme badges to same line
  ([`125fa6a`](https://github.com/sradc/SmallPebble/commit/125fa6a711074945ccdfc288f589905277fad914))

- Readme tweaks, explain low cnn accuracy due to CPU time, add dates to footer
  ([`155310b`](https://github.com/sradc/SmallPebble/commit/155310bc59b82d68889bf8bad2e9a439f01547d3))

### Testing

- Restore the tensorflow tests, dependency in dev group
  ([`164522a`](https://github.com/sradc/SmallPebble/commit/164522a9816aa84918dd4b5cc686fdb7094ba44e))


## v3.0.1 (2026-01-18)

### Bug Fixes

- Trigger new release
  ([`8f9f32e`](https://github.com/sradc/SmallPebble/commit/8f9f32e34e99c58212495d71d20d2706241caae8))

### Continuous Integration

- Only push to pypi if dist exists
  ([`d6ba9bb`](https://github.com/sradc/SmallPebble/commit/d6ba9bb8f655bcac4a02f86e4b5caf41b4451386))


## v3.0.0 (2026-01-18)

- Initial Release
