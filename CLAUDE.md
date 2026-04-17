# CLAUDE.md

Personal blog for Karimi Zayan, deployed at https://karimizayan.com. Based on the AstroPaper theme.

## Stack
- **Astro 5** with content collections
- **pnpm 10.11.1** (the repo has `pnpm-lock.yaml` — never run `npm install`)
- **Tailwind 4** for styling, **KaTeX** for math rendering
- **Deployed** via GitHub Actions to GitHub Pages from `waffleSheep/tough_blog`

## Where things live
- `src/data/blog/*.md` — blog posts. Frontmatter schema is in `src/content.config.ts`.
- `src/pages/index.astro` — homepage (hero, featured, recent posts).
- `src/pages/about.md` — About page body.
- `src/pages/posts/`, `src/pages/tags/` — routing; usually don't touch.
- `src/config.ts` — site-wide settings: title, author, `SITE.website`, theme toggle, post-per-page, etc.
- `src/constants.ts` — `SOCIALS` array (GitHub + LinkedIn icons in the footer).
- `src/components/Header.astro`, `Footer.astro`, `Socials.astro` — nav + footer layout.
- `src/layouts/` — page layouts; `Layout.astro` is the root HTML layout and imports `katex/dist/katex.min.css` globally.
- `astro.config.ts` — markdown plugins, Vite settings, sitemap.
- `public/CNAME` — custom domain pin for GitHub Pages. Do not remove.
- `.github/workflows/deploy.yml` — the deploy pipeline. Only edit if deploy strategy changes.

## Content conventions

### Blog post frontmatter
Every post in `src/data/blog/` must have this frontmatter (see `src/content.config.ts` for the full schema):

```yaml
---
author: Karimi Zayan
pubDatetime: 2026-04-17T12:00:00Z
title: Post title
slug: post-title-kebab-case
featured: false            # true → shows in homepage Featured section
draft: false               # true → excluded from build
tags:
  - machine-learning
description: One-line summary used in meta tags and post lists.
---
```

- Filename (kebab-case `.md`) doubles as the URL slug if `slug` is omitted. Keep `slug` explicit for stability.
- `pubDatetime` must be an ISO-8601 UTC string.
- Subject matter is mainly **math and computer science** — expect LaTeX math and code blocks in most posts.

### LaTeX / math
- Inline math: `$E = mc^2$` → $E = mc^2$
- Display math: wrap in `$$ ... $$` on its own lines.
- Plugins (`remark-math` + `rehype-katex`) are registered in `astro.config.ts`. Don't re-register them in frontmatter or per-post configs.
- KaTeX CSS is loaded once globally in `Layout.astro` — don't add `<link>` tags to individual posts.

### Code blocks
- Shiki is configured with themes `min-light` (light mode) and `night-owl` (dark mode), plus notation transformers (`// [!code highlight]`, `// [!code ++]`, etc. work inside fenced blocks).

## Common edit tasks

- **Change site title / author / description** → `src/config.ts` `SITE` object.
- **Add or change social link** → `src/constants.ts` `SOCIALS` array.
- **Toggle nav items** → `src/components/Header.astro`. Archives is gated on `SITE.showArchives`.
- **Change homepage hero** → `src/pages/index.astro`, `#hero` section.
- **Add a new blog post** → create a `.md` file in `src/data/blog/` with the frontmatter above.
- **Re-enable "Edit page" button on posts** → `SITE.editPost.enabled = true` in `src/config.ts`.

## Local dev

- **Start dev server**: `pnpm dev --host` — binds to 0.0.0.0, reachable at http://localhost:4321/ from Windows. The `--host` flag matters for WSL2.
- **HMR on WSL2**: polling is enabled in `astro.config.ts` (`vite.server.watch.usePolling`). Do not remove this — HMR does not work on `/mnt/c` without it. `node_modules` is excluded from the watcher; without that exclusion, startup hangs for over a minute.
- **Build**: `pnpm build` — runs `astro check`, builds, generates Pagefind index. Expect ~60s on first build.
- **Preview prod build**: `pnpm preview`.
- **Lint / format**: `pnpm lint`, `pnpm format`, `pnpm format:check`.

## Deploy
- `main` branch → `.github/workflows/deploy.yml` → GitHub Pages.
- Every push to `main` triggers a deploy. There's no staging.
- Custom domain (`karimizayan.com`) is configured via `public/CNAME` + the GitHub Pages API. DNS is apex A records to GitHub's `185.199.108-111.153`.
- `.github/workflows/ci.yml` runs on PRs only (lint + format + build check).

## Don'ts
- **Don't commit `package-lock.json` or `yarn.lock`** — pnpm is the source of truth.
- **Don't remove `public/CNAME`** — removing it un-sets the custom domain on next deploy.
- **Don't disable polling** in `astro.config.ts` while the project lives on `/mnt/c`.
- **Don't write markdown files outside `src/data/blog/` and expect them to become posts** — only that folder is a content collection.
- **Don't add tracking scripts or analytics** without asking first.
