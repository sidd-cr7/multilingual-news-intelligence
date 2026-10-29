# Changes in this delivery

- Fixed a frontend submission lock bug: invalid text/URL/PDF input previously returned after setting `busy.current = true`, which could prevent subsequent valid submissions for the lifetime of the page. The lock is now acquired only after input validation passes.
- Added a root README with Windows setup, run, test, and evaluation instructions.
- Removed generated `.next` build/cache output, Python bytecode/cache folders, pytest cache, and the local `frontend/.env.local` file from the delivered archive. Source files, dependency manifests/lockfile, tests, evaluation fixtures, architecture documentation, and `.env.local.example` are retained.
- No NLP model or API behavior has been represented as newly implemented without evidence. The existing analysis engine remains explicitly documented as heuristic/rule-based, not a validated article-level bias classifier.
