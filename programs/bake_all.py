#!/usr/bin/env python3
"""
bake_all.py — Script Blender (exécution en mode background)

Objectif :
- Forcer Blender à écrire TOUS les caches dans un répertoire unique (--cache-dir)
- Exploiter au maximum les threads CPU via OpenMP (Mantaflow) et mode FIXED
- Produire un cache_manifest.json pour validation par le pipeline
- Supporte la reprise : NE PAS supprimer les caches existants par défaut

Interface :
  blender --background fichier.blend --python bake_all.py -- \
    --cache-dir /path/to/cache [options]
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bpy

# ═══════════════════════════════════════════
# Constantes
# ═══════════════════════════════════════════

CACHE_SUBDIRS = ("ptcache", "fluids", "rigidbody", "alembic", "geonodes")
RESERVE_THREADS = 2

# Extensions de cache attendues par le pipeline (pipeline.py / FrameWatcher)
CACHE_EXTENSIONS = {'.bphys', '.vdb', '.uni', '.gz', '.png', '.exr', '.abc', '.obj', '.ply'}

# ═══════════════════════════════════════════
# État global interruption
# ═══════════════════════════════════════════

_interrupted = False
_interrupt_count = 0


# ═══════════════════════════════════════════
# Logging (stdout uniquement — lu par blender_runner.py)
# ═══════════════════════════════════════════

def log(msg: str) -> None:
    print(f"[BAKE_ALL] {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"[BAKE_ALL][WARN] {msg}", flush=True)


def err(msg: str) -> None:
    print(f"[BAKE_ALL][ERROR] {msg}", flush=True)


# ═══════════════════════════════════════════
# Signal handlers
# ═══════════════════════════════════════════

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


# ═══════════════════════════════════════════
# Parsing arguments
# ═══════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender bake-all helper")

    parser.add_argument("--cache-dir", required=True,
                        help="Répertoire racine des caches")

    parser.add_argument("--frame-start", "--start-frame",
                        dest="frame_start", type=int, default=None)
    parser.add_argument("--frame-end", "--end-frame",
                        dest="frame_end", type=int, default=None)

    parser.add_argument("--clear-existing", action="store_true",
                        help="Supprime les caches avant bake (DANGEREUX)")

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


# ═══════════════════════════════════════════
# Vérifications
# ═══════════════════════════════════════════

def verify_blend_loaded() -> bool:
    if not bpy.data.filepath:
        err("Aucun fichier .blend chargé")
        return False
    log(f"Fichier .blend chargé : {bpy.data.filepath}")
    return True


# ═══════════════════════════════════════════
# Création répertoires de cache
# ═══════════════════════════════════════════

def setup_cache_directories(cache_root: Path) -> Dict[str, Path]:
    dirs: Dict[str, Path] = {}
    for name in CACHE_SUBDIRS:
        d = cache_root / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs


# ═══════════════════════════════════════════
# Symlink ptcache (fallback pour Blender qui écrit dans blendcache_<stem>)
# ═══════════════════════════════════════════

def setup_ptcache_symlink(cache_root: Path) -> bool:
    """
    Crée un symlink blendcache_<stem> → <cache_root>/ptcache/
    Blender écrit les ptcache dans blendcache_<blend_stem>/ à côté du .blend.
    Le symlink redirige vers notre répertoire de cache unifié.
    """
    blend_path = Path(bpy.data.filepath)
    if not blend_path.exists():
        return False

    blendcache_dir = blend_path.parent / f"blendcache_{blend_path.stem}"
    target = cache_root / "ptcache"

    # Vérifier si le symlink existe déjà et pointe au bon endroit
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

    # Supprimer si c'est un vrai dossier
    if blendcache_dir.is_dir() and not blendcache_dir.is_symlink():
        try:
            shutil.rmtree(str(blendcache_dir), ignore_errors=True)
        except OSError:
            return False

    # Supprimer si c'est un fichier
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


# ═══════════════════════════════════════════
# Configuration threading
# ═══════════════════════════════════════════

def configure_threading(scene: bpy.types.Scene, n_threads: int) -> None:
    """
    Configure le threading Blender.
    Note : OMP_NUM_THREADS est déjà défini par blender_runner.py dans l'env
    du subprocess AVANT le lancement. On le remet ici par sécurité mais
    OpenMP l'a déjà lu au démarrage du processus.
    """
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ.pop("OMP_PROC_BIND", None)

    try:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = n_threads
    except Exception:
        pass

    log(f"Threading configuré : {n_threads} threads, mode=FIXED")


# ═══════════════════════════════════════════
# Configuration des caches — Fluid Domains
# ═══════════════════════════════════════════

def configure_fluid_domains(scene: bpy.types.Scene, fluids_dir: Path) -> int:
    """Redirige tous les fluid domains vers fluids_dir."""
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
                log(f"  Fluid domain '{obj.name}' → {fluids_dir}")
            except Exception as e:
                warn(f"Erreur config fluid '{obj.name}' : {e}")
    return count


# ═══════════════════════════════════════════
# Configuration des caches — Point Caches (ptcache)
# ═══════════════════════════════════════════

def _configure_single_point_cache(pc: Any, ptcache_dir: Path) -> bool:
    """Configure un point_cache individuel pour écriture disque."""
    try:
        if hasattr(pc, "use_disk_cache"):
            pc.use_disk_cache = True
        if hasattr(pc, "use_external"):
            pc.use_external = False
        if hasattr(pc, "use_library_path"):
            pc.use_library_path = False
        # Redirection du chemin : Blender 3.x utilise filepath_raw
        # pour les point caches quand use_external est False,
        # le cache va dans blendcache_<stem>/ (géré par le symlink)
        return True
    except Exception as e:
        warn(f"Erreur configuration point_cache : {e}")
        return False


def configure_disk_caches(scene: bpy.types.Scene, ptcache_dir: Path) -> int:
    """Configure tous les point caches de la scène pour écriture disque."""
    count = 0

    # Rigid Body World
    rbw = getattr(scene, "rigidbody_world", None)
    if rbw and rbw.point_cache:
        if _configure_single_point_cache(rbw.point_cache, ptcache_dir):
            count += 1

    for obj in scene.objects:
        # Particle Systems
        for psys in getattr(obj, "particle_systems", []):
            if psys.point_cache:
                if _configure_single_point_cache(psys.point_cache, ptcache_dir):
                    count += 1

        # Modifiers avec point_cache
        for mod in obj.modifiers:
            if mod.type in ("CLOTH", "SOFT_BODY"):
                if mod.point_cache:
                    if _configure_single_point_cache(mod.point_cache, ptcache_dir):
                        count += 1
            elif mod.type == "DYNAMIC_PAINT":
                canvas = getattr(mod, "canvas_settings", None)
                if canvas and hasattr(canvas, "canvas_surfaces"):
                    for surf in canvas.canvas_surfaces:
                        if surf.point_cache:
                            if _configure_single_point_cache(surf.point_cache, ptcache_dir):
                                count += 1
            elif mod.type != "FLUID" and hasattr(mod, "point_cache") and mod.point_cache:
                if _configure_single_point_cache(mod.point_cache, ptcache_dir):
                    count += 1

    log(f"  {count} caches disque configurés")
    return count


# ═══════════════════════════════════════════
# Clear caches existants
# ═══════════════════════════════════════════

def clear_all_caches(scene: bpy.types.Scene) -> None:
    """Supprime tous les bakes existants."""
    try:
        bpy.ops.ptcache.free_bake_all()
        log("  ptcache.free_bake_all() → OK")
    except Exception as e:
        warn(f"  ptcache.free_bake_all() échoué : {e}")

    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type == "FLUID" and getattr(mod, "fluid_type", None) == "DOMAIN":
                try:
                    bpy.context.view_layer.objects.active = obj
                    bpy.ops.fluid.free_all()
                    log(f"  fluid.free_all() → OK ({obj.name})")
                except Exception as e:
                    warn(f"  fluid.free_all() échoué ({obj.name}) : {e}")


# ═══════════════════════════════════════════
# Bake — Point Caches (individuel par objet, plus robuste en background)
# ═══════════════════════════════════════════

def _ensure_context(scene: bpy.types.Scene, obj: bpy.types.Object) -> bool:
    """Configure le contexte Blender pour un objet (requis par bpy.ops en background)."""
    try:
        bpy.context.window.scene = scene
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        return True
    except Exception as e:
        warn(f"Impossible de configurer le contexte pour '{obj.name}' : {e}")
        return False


def bake_point_caches_individual(scene: bpy.types.Scene) -> Tuple[int, int]:
    """
    Bake les point caches un par un (plus robuste que bake_all en background).
    Retourne (succès, échecs).
    """
    if _interrupted:
        return 0, 0

    successes = 0
    failures = 0

    # D'abord essayer bake_all (fonctionne pour la plupart des cas)
    try:
        log(f"  ptcache.bake_all(bake=True)...")
        bpy.ops.ptcache.bake_all(bake=True)
        # Compter les caches qui ont été baked
        for obj in scene.objects:
            for psys in getattr(obj, "particle_systems", []):
                if psys.point_cache and psys.point_cache.is_baked:
                    successes += 1
            for mod in obj.modifiers:
                if hasattr(mod, "point_cache") and mod.point_cache:
                    if mod.point_cache.is_baked:
                        successes += 1
        rbw = getattr(scene, "rigidbody_world", None)
        if rbw and rbw.point_cache and rbw.point_cache.is_baked:
            successes += 1

        log(f"  ptcache.bake_all → {successes} caches baked")
        return successes, failures

    except Exception as e:
        warn(f"  ptcache.bake_all échoué : {e}")
        warn(f"  Fallback : bake individuel par objet...")

    # Fallback : bake individuel
    # Rigid Body World
    rbw = getattr(scene, "rigidbody_world", None)
    if rbw and rbw.point_cache:
        if _interrupted:
            return successes, failures
        try:
            pc = rbw.point_cache
            if not pc.is_baked:
                # Pour rigid body, on utilise bake_all car il n'y a pas
                # d'opérateur individuel simple
                bpy.ops.ptcache.bake_all(bake=True)
                successes += 1
                log(f"  Rigid Body World → baked")
        except Exception as e:
            failures += 1
            warn(f"  Rigid Body World → échec : {e}")

    for obj in scene.objects:
        if _interrupted:
            return successes, failures

        # Particle Systems
        for i, psys in enumerate(getattr(obj, "particle_systems", [])):
            if _interrupted:
                return successes, failures
            if not psys.point_cache or psys.point_cache.is_baked:
                continue
            try:
                if _ensure_context(scene, obj):
                    # Sélectionner le bon particle system index
                    obj.particle_systems.active_index = i
                    bpy.ops.ptcache.bake({"point_cache": psys.point_cache}, bake=True)
                    successes += 1
                    log(f"  Particules '{obj.name}' [{i}] → baked")
            except Exception as e:
                failures += 1
                warn(f"  Particules '{obj.name}' [{i}] → échec : {e}")

        # Modifiers avec point_cache (Cloth, SoftBody, Dynamic Paint)
        for mod in obj.modifiers:
            if _interrupted:
                return successes, failures
            if mod.type == "FLUID":
                continue  # Géré séparément
            pc = getattr(mod, "point_cache", None)
            if not pc or pc.is_baked:
                continue
            try:
                if _ensure_context(scene, obj):
                    bpy.ops.ptcache.bake({"point_cache": pc}, bake=True)
                    successes += 1
                    log(f"  {mod.type} '{obj.name}.{mod.name}' → baked")
            except Exception as e:
                failures += 1
                warn(f"  {mod.type} '{obj.name}.{mod.name}' → échec : {e}")

            # Dynamic Paint surfaces
            if mod.type == "DYNAMIC_PAINT":
                canvas = getattr(mod, "canvas_settings", None)
                if canvas and hasattr(canvas, "canvas_surfaces"):
                    for surf in canvas.canvas_surfaces:
                        if _interrupted:
                            return successes, failures
                        spc = getattr(surf, "point_cache", None)
                        if not spc or spc.is_baked:
                            continue
                        try:
                            if _ensure_context(scene, obj):
                                bpy.ops.ptcache.bake({"point_cache": spc}, bake=True)
                                successes += 1
                                log(f"  DynamicPaint surface '{obj.name}' → baked")
                        except Exception as e:
                            failures += 1
                            warn(f"  DynamicPaint surface '{obj.name}' → échec : {e}")

    return successes, failures


# ═══════════════════════════════════════════
# Bake — Fluid Domains (Mantaflow)
# ═══════════════════════════════════════════

def bake_fluid_domains(scene: bpy.types.Scene) -> Tuple[int, int]:
    """Bake tous les fluid domains. Retourne (succès, échecs)."""
    successes = 0
    failures = 0

    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type != "FLUID" or getattr(mod, "fluid_type", None) != "DOMAIN":
                continue
            if _interrupted:
                return successes, failures
            try:
                if _ensure_context(scene, obj):
                    bpy.ops.fluid.bake_all()
                    successes += 1
                    log(f"  Fluid domain '{obj.name}' → baked")
            except Exception as e:
                failures += 1
                warn(f"  Fluid domain '{obj.name}' → échec : {e}")

    return successes, failures


# ═══════════════════════════════════════════
# Manifest de cache
# ═══════════════════════════════════════════

def collect_cache_files(cache_root: Path) -> List[Dict[str, Any]]:
    """Collecte tous les fichiers de cache avec métadonnées."""
    files = []
    for f in sorted(cache_root.rglob("*")):
        if not f.is_file():
            continue
        if f.suffix.lower() not in CACHE_EXTENSIONS:
            continue
        if f.name == "cache_manifest.json":
            continue
        try:
            stat = f.stat()
            files.append({
                "path": str(f.relative_to(cache_root)),
                "size": stat.st_size,
                "timestamp": datetime.datetime.fromtimestamp(
                    stat.st_mtime
                ).isoformat(),
            })
        except OSError:
            pass
    return files


def write_manifest(
    cache_root: Path,
    scene_name: str,
    frame_start: int,
    frame_end: int,
    status: str,
    errors: List[str],
    duration: float,
    bake_stats: Dict[str, int],
) -> None:
    """Écrit le cache_manifest.json."""
    cache_files = collect_cache_files(cache_root)

    total_size = sum(f["size"] for f in cache_files)

    manifest = {
        "blender_version": bpy.app.version_string,
        "scene": scene_name,
        "frame_range": [frame_start, frame_end],
        "cache_dir": str(cache_root),
        "timestamp": datetime.datetime.now().isoformat(),
        "duration_seconds": round(duration, 2),
        "status": status,
        "bake_stats": bake_stats,
        "errors": errors,
        "total_cache_size": total_size,
        "file_count": len(cache_files),
        "files": cache_files,
    }

    manifest_path = cache_root / "cache_manifest.json"
    try:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log(f"Manifest écrit : {manifest_path} ({len(cache_files)} fichiers, {total_size} octets)")
    except Exception as e:
        warn(f"Impossible d'écrire le manifest : {e}")


# ═══════════════════════════════════════════
# Point d'entrée principal
# ═══════════════════════════════════════════

def main() -> int:
    install_signal_handlers()
    args = parse_args()

    start_time = time.time()

    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    cpu_count = os.cpu_count() or 1
    n_threads = args.bake_threads if args.bake_threads else max(1, cpu_count - RESERVE_THREADS)

    # ── Bannière de démarrage ──
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

    # ── Créer les sous-répertoires de cache ──
    cache_dirs = setup_cache_directories(cache_root)
    setup_ptcache_symlink(cache_root)

    # ── Scènes à traiter ──
    scenes = list(bpy.data.scenes) if args.all_scenes else [bpy.context.scene]

    # ── Statistiques globales ──
    all_errors: List[str] = []
    total_successes = 0
    total_failures = 0
    final_status = "complete"
    last_scene_name = ""
    frame_start = 1
    frame_end = 250

    for scene in scenes:
        if _interrupted:
            break

        last_scene_name = scene.name
        log(f"─── Scène : {scene.name} ───")

        try:
            # Threading
            configure_threading(scene, n_threads)

            # Frame range
            if args.frame_start is not None:
                scene.frame_start = args.frame_start
            if args.frame_end is not None:
                scene.frame_end = args.frame_end
            frame_start = scene.frame_start
            frame_end = scene.frame_end
            log(f"  Frame range : {frame_start} → {frame_end}")

            # Configurer les caches disque
            configure_disk_caches(scene, cache_dirs["ptcache"])

            # Configurer les fluid domains
            if args.bake_fluids:
                n_fluids = configure_fluid_domains(scene, cache_dirs["fluids"])
                log(f"  {n_fluids} fluid domain(s) configuré(s)")

            # Clear si demandé
            if args.clear_existing:
                warn("clear-existing activé : suppression des caches existants")
                clear_all_caches(scene)

            # ── Bake Point Caches ──
            if args.bake_cloth or args.bake_particles:
                log(f"[{scene.name}] Bake point caches…")
                pc_ok, pc_fail = bake_point_caches_individual(scene)
                total_successes += pc_ok
                total_failures += pc_fail
                if pc_fail > 0:
                    all_errors.append(f"[{scene.name}] {pc_fail} point cache(s) échoué(s)")

            # ── Bake Fluids ──
            if args.bake_fluids and not _interrupted:
                log(f"[{scene.name}] Bake fluid domains…")
                fl_ok, fl_fail = bake_fluid_domains(scene)
                total_successes += fl_ok
                total_failures += fl_fail
                if fl_fail > 0:
                    all_errors.append(f"[{scene.name}] {fl_fail} fluid domain(s) échoué(s)")

        except Exception as e:
            error_msg = f"[{scene.name}] Erreur inattendue : {e}"
            err(error_msg)
            all_errors.append(error_msg)
            total_failures += 1

    # ── Déterminer le statut final ──
    duration = time.time() - start_time

    if _interrupted:
        final_status = "interrupted"
    elif total_failures > 0 and total_successes > 0:
        final_status = "partial"
    elif total_failures > 0 and total_successes == 0:
        final_status = "failed"
    else:
        final_status = "complete"

    # ── Écrire le manifest ──
    bake_stats = {
        "successes": total_successes,
        "failures": total_failures,
        "total": total_successes + total_failures,
    }

    write_manifest(
        cache_root=cache_root,
        scene_name=last_scene_name or "unknown",
        frame_start=frame_start,
        frame_end=frame_end,
        status=final_status,
        errors=all_errors,
        duration=duration,
        bake_stats=bake_stats,
    )

    # ── Résumé final ──
    cache_files = collect_cache_files(cache_root)
    total_size = sum(f["size"] for f in cache_files)

    log("=" * 70)
    log(f"RÉSUMÉ — statut: {final_status.upper()}")
    log(f"  Durée          : {duration:.1f}s")
    log(f"  Bakes réussis  : {total_successes}")
    log(f"  Bakes échoués  : {total_failures}")
    log(f"  Fichiers cache : {len(cache_files)}")
    log(f"  Taille totale  : {total_size} octets")
    if all_errors:
        log(f"  Erreurs :")
        for e in all_errors:
            log(f"    - {e}")
    log("=" * 70)

    # ── Code de sortie ──
    if _interrupted:
        return 1
    if total_failures > 0 and total_successes == 0:
        return 1
    if total_failures > 0 and total_successes > 0:
        if args.strict:
            return 1
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())