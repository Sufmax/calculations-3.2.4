#!/usr/bin/env python3
"""
bake_all.py — Script Blender (exécution en mode background)

Objectif :
- Forcer Blender à écrire TOUS les caches dans un répertoire unique (--cache-dir)
- Exploiter au maximum les threads CPU via OpenMP (Mantaflow) et mode FIXED
- Supporte la reprise : NE PAS supprimer les caches existants par défaut

Interface attendue :
  blender --background fichier.blend --python bake_all.py -- --start-frame 139 --end-frame 250 --cache-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Dict

import bpy

CACHE_SUBDIRS = ("ptcache", "fluids", "rigidbody", "alembic", "geonodes")
RESERVE_THREADS = 2

_interrupted = False
_interrupt_count = 0


def log(msg: str) -> None:
    print(f"[BAKE_ALL] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[BAKE_ALL][WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[BAKE_ALL][ERROR] {msg}", flush=True)


def _signal_handler(signum: int, frame: Any) -> None:
    global _interrupted, _interrupt_count
    _interrupted = True
    _interrupt_count += 1
    sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
    warn(f"Signal {sig_name} reçu (#{_interrupt_count})")
    if _interrupt_count >= 3:
        err("3 interruptions → arrêt immédiat")
        sys.exit(1)


def install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _signal_handler)
        except (OSError, ValueError):
            pass


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender bake-all helper")

    parser.add_argument("--cache-dir", required=True, help="Répertoire racine des caches")

    # Aliases start/end frame (compat spec)
    parser.add_argument("--frame-start", "--start-frame", dest="frame_start", type=int, default=None)
    parser.add_argument("--frame-end", "--end-frame", dest="frame_end", type=int, default=None)

    parser.add_argument("--clear-existing", action="store_true", help="Supprime les caches avant bake (DANGEREUX)")
    parser.add_argument("--progress-file", type=str, default=None, help="Fichier JSON de statut global")

    parser.add_argument("--bake-fluids", action="store_true", default=True)
    parser.add_argument("--no-bake-fluids", dest="bake_fluids", action="store_false")
    parser.add_argument("--bake-particles", action="store_true", default=True)
    parser.add_argument("--no-bake-particles", dest="bake_particles", action="store_false")
    parser.add_argument("--bake-cloth", action="store_true", default=True)
    parser.add_argument("--no-bake-cloth", dest="bake_cloth", action="store_false")

    parser.add_argument("--bake-threads", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--all-scenes", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args(argv)


def verify_blend_loaded() -> bool:
    if not bpy.data.filepath:
        err("Aucun fichier .blend chargé")
        return False
    log(f"Fichier .blend chargé : {bpy.data.filepath}")
    return True


def setup_cache_directories(cache_root: Path) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for name in CACHE_SUBDIRS:
        d = cache_root / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


def setup_ptcache_symlink(cache_root: Path, verbose: bool = False) -> bool:
    blend_path = Path(bpy.data.filepath)
    if not blend_path.exists():
        return False

    blendcache_dir = blend_path.parent / f"blendcache_{blend_path.stem}"
    target = cache_root / "ptcache"

    if blendcache_dir.is_symlink():
        try:
            if blendcache_dir.resolve() == target.resolve():
                return True
        except OSError:
            pass
        try:
            blendcache_dir.unlink()
        except OSError:
            return False

    if blendcache_dir.is_dir() and not blendcache_dir.is_symlink():
        try:
            shutil.rmtree(str(blendcache_dir), ignore_errors=True)
        except OSError:
            return False

    if blendcache_dir.exists() and not blendcache_dir.is_dir():
        try:
            blendcache_dir.unlink()
        except OSError:
            return False

    try:
        blendcache_dir.symlink_to(target, target_is_directory=True)
        log(f"Symlink créé : {blendcache_dir} → {target}")
        return True
    except OSError as e:
        warn(f"Échec symlink : {e}")
        return False


def configure_threading(scene: bpy.types.Scene, n_threads: int, verbose: bool = False) -> None:
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.pop("OMP_PROC_BIND", None)

    try:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = n_threads
    except Exception:
        pass

    if verbose:
        log(f"Threading : {n_threads} threads, mode=FIXED, OMP_NUM_THREADS={n_threads}")


def configure_fluid_domains(scene: bpy.types.Scene, fluids_dir: Path, verbose: bool = False) -> int:
    count = 0
    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type != "FLUID":
                continue
            if getattr(mod, "fluid_type", None) != "DOMAIN":
                continue
            ds = getattr(mod, "domain_settings", None)
            if ds is None:
                continue
            try:
                ds.cache_directory = str(fluids_dir)
                if hasattr(ds, "cache_data_format"):
                    ds.cache_data_format = "OPENVDB"
                if hasattr(ds, "openvdb_cache_compress_type"):
                    ds.openvdb_cache_compress_type = "BLOSC"
                count += 1
                if verbose:
                    log(f"Fluid domain '{obj.name}' → {fluids_dir}")
            except Exception as e:
                warn(f"Erreur config fluid '{obj.name}' : {e}")
    return count


def configure_disk_caches(scene: bpy.types.Scene, verbose: bool = False) -> int:
    count = 0

    def _configure_pc(pc: Any) -> bool:
        nonlocal count
        try:
            if hasattr(pc, "use_disk_cache"):
                pc.use_disk_cache = True
            if hasattr(pc, "use_external"):
                pc.use_external = False
            if hasattr(pc, "use_library_path"):
                pc.use_library_path = False
            count += 1
            return True
        except Exception:
            return False

    rbw = getattr(scene, "rigidbody_world", None)
    if rbw and rbw.point_cache:
        _configure_pc(rbw.point_cache)

    for obj in scene.objects:
        for psys in getattr(obj, "particle_systems", []):
            if psys.point_cache:
                _configure_pc(psys.point_cache)

        for mod in obj.modifiers:
            if mod.type in ("CLOTH", "SOFT_BODY"):
                if mod.point_cache:
                    _configure_pc(mod.point_cache)
            elif mod.type == "DYNAMIC_PAINT":
                canvas = getattr(mod, "canvas_settings", None)
                if canvas and hasattr(canvas, "canvas_surfaces"):
                    for surf in canvas.canvas_surfaces:
                        if surf.point_cache:
                            _configure_pc(surf.point_cache)
            elif mod.type != "FLUID" and hasattr(mod, "point_cache") and mod.point_cache:
                _configure_pc(mod.point_cache)

    if verbose:
        log(f"Total caches disque configurés : {count}")
    return count


def clear_all_caches(scene: bpy.types.Scene, verbose: bool = False) -> None:
    try:
        bpy.ops.ptcache.free_bake_all()
    except Exception:
        pass

    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type == "FLUID" and getattr(mod, "fluid_type", None) == "DOMAIN":
                try:
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.fluid.free_all()
                except Exception:
                    pass


def bake_point_caches(scene: bpy.types.Scene, verbose: bool = False) -> bool:
    if _interrupted:
        return False
    try:
        bpy.ops.ptcache.bake_all(bake=True)
        if verbose:
            log("ptcache.bake_all(bake=True) → OK")
        return True
    except Exception as e:
        warn(f"ptcache.bake_all échoué : {e}")
        return False


def bake_fluid_domains(scene: bpy.types.Scene, verbose: bool = False) -> bool:
    ok = True
    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type != "FLUID" or getattr(mod, "fluid_type", None) != "DOMAIN":
                continue
            if _interrupted:
                return False
            try:
                bpy.context.view_layer.objects.active = obj
                bpy.ops.fluid.bake_all()
                if verbose:
                    log(f"fluid.bake_all → OK ({obj.name})")
            except Exception as e:
                ok = False
                warn(f"fluid bake '{obj.name}' : {e}")
    return ok


def write_progress(progress_file: str | None, payload: Dict[str, Any]) -> None:
    if not progress_file:
        return
    try:
        path = Path(progress_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    except Exception as e:
        warn(f"Impossible d'écrire progress-file: {e}")


def main() -> int:
    install_signal_handlers()
    args = parse_args()

    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    n_threads = args.bake_threads if args.bake_threads else max(1, cpu_count - RESERVE_THREADS)

    log("=" * 70)
    log("Démarrage bake_all.py")
    log(f"  Fichier .blend: {bpy.data.filepath}")
    log(f"  Cache dir     : {cache_root}")
    log(f"  CPU           : {cpu_count} threads")
    log(f"  Bake threads  : {n_threads}")
    log(f"  Frame start   : {args.frame_start}")
    log(f"  Frame end     : {args.frame_end}")
    log("=" * 70)

    if not verify_blend_loaded():
        return 1

    write_progress(args.progress_file, {
        "status": "running",
        "startedAt": datetime.datetime.now().isoformat(),
        "frameStart": args.frame_start,
        "frameEnd": args.frame_end,
        "cacheDir": str(cache_root),
    })

    cache_dirs = setup_cache_directories(cache_root)
    setup_ptcache_symlink(cache_root, verbose=args.verbose)

    scenes = list(bpy.data.scenes) if args.all_scenes else [bpy.context.scene]

    for scene in scenes:
        if _interrupted:
            break

        try:
            # Appliquer configuration
            configure_threading(scene, n_threads, verbose=args.verbose)

            if args.frame_start:
                scene.frame_start = args.frame_start
            if args.frame_end:
                scene.frame_end = args.frame_end

            configure_disk_caches(scene, verbose=args.verbose)

            if args.bake_fluids:
                configure_fluid_domains(scene, cache_dirs["fluids"], verbose=args.verbose)

            # IMPORTANT : on ne supprime les caches existants que si explicitement demandé
            if args.clear_existing:
                warn("clear-existing activé : suppression des caches existants")
                clear_all_caches(scene, verbose=args.verbose)

            # Bake
            if args.bake_cloth or args.bake_particles:
                log(f"[{scene.name}] Bake point caches…")
                bake_point_caches(scene, verbose=args.verbose)

            if args.bake_fluids and not _interrupted:
                log(f"[{scene.name}] Bake fluid domains…")
                bake_fluid_domains(scene, verbose=args.verbose)

        except Exception as e:
            err(f"[{scene.name}] Erreur : {e}")

    write_progress(args.progress_file, {
        "status": "finished" if not _interrupted else "interrupted",
        "finishedAt": datetime.datetime.now().isoformat(),
    })

    if _interrupted:
        log("RÉSUMÉ — statut: INTERRUPTED")
        return 1

    log("BAKE COMPLET — succès")
    return 0


if __name__ == "__main__":
    sys.exit(main())