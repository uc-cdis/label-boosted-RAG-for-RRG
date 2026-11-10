# Data audit tools

Each node keeps its own copy of `/opt/gpudata/cxr`, and there is no shared filesystem. These
scripts check that every copy matches the reference node, **kg35-nvl01**, whose files were
verified against the dataset authors' checksums ([docs/data.md](../../docs/data.md)). They
need only bash, coreutils and python3 >= 3.8, not the conda env. Run them on the host.

## On each node

```bash
# 1. Move the locally derived files into <dataset>/derived/ (dry run first).
tools/data-audit/segregate.sh /opt/gpudata/cxr
tools/data-audit/segregate.sh /opt/gpudata/cxr --apply

# 2. Hash every file; about 20 minutes for the 3.2 TB at ~3 GB/s.
tools/data-audit/manifest.sh /opt/gpudata/cxr ~/data-audit/$(hostname -s)

# 3. Compare the per-dataset digests with the reference node.
diff tools/data-audit/kg35-nvl01.digests.txt ~/data-audit/$(hostname -s)/digests.txt
```

If a digest differs, copy the reference manifests from kg35-nvl01
(`/home/songs1/labrag-outputs/data-audit/reference-kg35-nvl01/`, 232 MB) and list the files:

```bash
python3 tools/data-audit/compare.py <reference dir> ~/data-audit/$(hostname -s)
```

## Updating the reference

After any change to kg35-nvl01's data (new author files, segregation), rerun step 2 there,
then copy its `digests.txt` to `kg35-nvl01.digests.txt` here and commit.
