#!/bin/bash
set -e

PRIVATE_DIR="${WIKI_PRIVATE_DIR:-../wiki-private}"
MANIFEST=".private-overlay-manifest"
EXCLUDE_MARKER="# BEGIN wiki-private overlay"
EXCLUDE_END="# END wiki-private overlay"

usage() {
    echo "Usage: $0 {link|unlink|status}"
    echo "  link   - Create symlinks from wiki-private into this repo"
    echo "  unlink - Remove all symlinks created by a previous link"
    echo "  status - Show what is currently linked"
    exit 1
}

do_unlink() {
    if [ ! -f "$MANIFEST" ]; then
        return 0
    fi

    local count=0
    while IFS= read -r relpath; do
        [ -z "$relpath" ] && continue
        if [ -L "$relpath" ]; then
            rm "$relpath"
            count=$((count + 1))
            # Remove parent directory if empty and not a top-level repo dir
            local dir
            dir="$(dirname "$relpath")"
            while [ "$dir" != "." ] && [ -d "$dir" ] && [ -z "$(ls -A "$dir" 2>/dev/null)" ]; do
                rmdir "$dir"
                dir="$(dirname "$dir")"
            done
        fi
    done < "$MANIFEST"

    rm -f "$MANIFEST"

    # Clean up .git/info/exclude
    if [ -f .git/info/exclude ]; then
        sed -i.bak "/$EXCLUDE_MARKER/,/$EXCLUDE_END/d" .git/info/exclude
        rm -f .git/info/exclude.bak
    fi

    echo "Overlay removed: $count symlinks deleted"
}

do_link() {
    if [ ! -d "$PRIVATE_DIR" ]; then
        echo "No private repo found at $PRIVATE_DIR, building public-only."
        exit 0
    fi

    # Idempotent: unlink first if manifest exists
    if [ -f "$MANIFEST" ]; then
        do_unlink
    fi

    local count=0
    local skipped=0
    : > "$MANIFEST"

    # Find all files in private repo, excluding .git and repo meta files
    while IFS= read -r filepath; do
        local relpath="${filepath#$PRIVATE_DIR/}"

        # Skip repo meta files
        case "$relpath" in
            .git/*|.gitignore|.gitattributes|README.md|README|LICENSE|CLAUDE.md) continue ;;
        esac

        # Check for conflicts with existing non-symlink files
        if [ -e "$relpath" ] && [ ! -L "$relpath" ]; then
            echo "WARNING: $relpath conflicts with existing file, skipping"
            skipped=$((skipped + 1))
            continue
        fi

        # Create parent directories if needed
        local dir
        dir="$(dirname "$relpath")"
        if [ "$dir" != "." ]; then
            mkdir -p "$dir"
        fi

        # Create symlink using absolute path to target
        ln -sf "$(cd "$PRIVATE_DIR" && pwd)/$relpath" "$relpath"
        echo "$relpath" >> "$MANIFEST"
        count=$((count + 1))
    done < <(find "$PRIVATE_DIR" -type f -not -path '*/.git/*')

    # Update .git/info/exclude
    if [ -f .git/info/exclude ]; then
        # Remove any previous overlay entries
        sed -i.bak "/$EXCLUDE_MARKER/,/$EXCLUDE_END/d" .git/info/exclude
        rm -f .git/info/exclude.bak
    fi
    mkdir -p .git/info
    {
        echo "$EXCLUDE_MARKER"
        cat "$MANIFEST"
        echo "$EXCLUDE_END"
    } >> .git/info/exclude

    echo "Overlay complete: $count files linked"
    if [ "$skipped" -gt 0 ]; then
        echo "  ($skipped files skipped due to conflicts)"
    fi
}

do_status() {
    if [ ! -f "$MANIFEST" ]; then
        echo "No overlay active"
        exit 0
    fi

    local total=0
    local valid=0
    local broken=0
    while IFS= read -r relpath; do
        [ -z "$relpath" ] && continue
        total=$((total + 1))
        if [ -L "$relpath" ] && [ -e "$relpath" ]; then
            echo "  OK: $relpath -> $(readlink "$relpath")"
            valid=$((valid + 1))
        elif [ -L "$relpath" ]; then
            echo "  BROKEN: $relpath -> $(readlink "$relpath")"
            broken=$((broken + 1))
        else
            echo "  MISSING: $relpath (not a symlink)"
            broken=$((broken + 1))
        fi
    done < "$MANIFEST"

    echo "$total linked ($valid valid, $broken broken)"
}

case "${1:-}" in
    link)   do_link ;;
    unlink) do_unlink ;;
    status) do_status ;;
    *)      usage ;;
esac
