# Manifest scope

`FULL_LOCAL_PACKAGE_MANIFEST.jsonl.gz` is the unchanged checksum manifest of the original complete local package. It includes omitted raw data and the earlier paper version. Its paper entries do not identify the revised public paper. This manifest is scientific provenance. It is not a manifest of the public release.

`OMITTED_LOCAL_FILES.jsonl.gz` lists saved scientific files excluded from the compact release. These files remain in the full local package.

`PUBLIC_FILE_MANIFEST.jsonl` at the release root identifies the actual public files. It excludes itself to avoid a circular checksum. Archive member manifests identify the exact decompressed metadata files. Both manifests preserve missing-file distinctions.

SHA-256 is a file-identity checksum. It detects changed bytes. It does not establish scientific validity.
