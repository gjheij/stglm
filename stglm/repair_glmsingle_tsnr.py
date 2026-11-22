"""Add PANIC-compatible tSNR names without invalidating existing provenance.

Run directly: python repair_glmsingle_tsnr.py ROOT [--apply]
Old names remain valid. Images are hard-linked, or copied if links are unsupported.
No BOLD/model arrays are loaded, and no saved output records are rewritten.
"""
import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


OLD = '_stat-tsnr_statmap.nii.gz'
NEW = '_stat-tsnr_boldmap.nii.gz'


def _digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def _fit_lock(folder):
    # Same lock used by fit_glmsingle, so an active export cannot be modified.
    with (folder.parent / ('.' + folder.name + '.fit.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _plan(folder):
    planned = []
    for source in sorted(folder.glob('qc/*' + OLD)):
        target = source.with_name(source.name[:-len(OLD)] + NEW)
        qc = source.with_name(source.name[:-len(OLD)] + '_qc.json')
        metadata = json.loads(qc.read_text())
        if not isinstance(metadata, dict):
            raise ValueError('QC sidecar must contain an object: ' + str(qc))
        # New image metadata uses its new name; existing sidecars remain intact.
        metadata['TSNRFile'] = str(target.resolve())
        sidecar = target.with_name(target.name[:-7] + '.json')
        if target.exists():
            if not os.path.samefile(source, target) and _digest(source) != _digest(target):
                raise ValueError('Conflicting destination image: ' + str(target))
        if sidecar.exists() and json.loads(sidecar.read_text()) != metadata:
            raise ValueError('Conflicting destination sidecar: ' + str(sidecar))
        planned.append((source, target, sidecar, metadata))
    return planned


def repair(root, apply=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError('Not a directory: ' + str(root))
    folders = sorted({path.parent.parent for path in root.rglob('*' + OLD) if path.parent.name == 'qc'})
    # Validate the entire selection before creating any compatibility outputs.
    for folder in folders:
        _plan(folder)
    runs = created = 0
    for folder in folders:
        with _fit_lock(folder):
            for source, target, sidecar, metadata in _plan(folder):
                runs += 1
                pending = not target.exists() or not sidecar.exists()
                print(('ADD ' if pending else 'OK  ') + str(target))
                if not apply or not pending:
                    continue
                if not target.exists():
                    try:
                        os.link(source, target)
                    except OSError as error:
                        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP, errno.EMLINK):
                            raise
                        # Exclusive destination creation; never replace a file.
                        created_target = False
                        try:
                            with source.open('rb') as src, target.open('xb') as dst:
                                created_target = True
                                shutil.copyfileobj(src, dst)
                            shutil.copystat(source, target)
                        except Exception:
                            if created_target:
                                target.unlink()
                            raise
                if not sidecar.exists():
                    descriptor, temporary = tempfile.mkstemp(prefix='.tsnr-repair-', dir=sidecar.parent)
                    try:
                        with os.fdopen(descriptor, 'w') as stream:
                            json.dump(metadata, stream, indent=2)
                            stream.write('\n')
                        os.link(temporary, sidecar)  # atomic publication without overwrite
                    finally:
                        Path(temporary).unlink(missing_ok=True)
                created += 1
    print(f'Runs={runs}; repaired={created}; mode={"apply" if apply else "preview"}; legacy files retained')
    return runs, created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='GLMsingle derivative root or one subject/stream directory')
    parser.add_argument('--apply', action='store_true', help='Create compatible names; default previews only')
    args = parser.parse_args()
    try:
        repair(args.root, args.apply)
    except (OSError, ValueError) as error:
        parser.exit(1, f'ERROR: {error}\n')


if __name__ == '__main__':
    main()
