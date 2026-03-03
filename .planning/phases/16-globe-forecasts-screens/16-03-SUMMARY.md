# Plan 16-03 Summary: Forecasts Screen

## Status: COMPLETE

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | SubmissionForm + SubmissionQueue components | `31f1b33` | SubmissionForm.ts, SubmissionQueue.ts, panels.css |
| 2 | Forecasts screen full rewrite | `e8af455` | forecasts-screen.ts |

## Deliverables

### SubmissionForm (`frontend/src/components/SubmissionForm.ts`)
- Three-state inline transform: input -> parsing -> confirm
- Input state: textarea with placeholder, "Analyze Question" submit button, validation
- Parsing state: CSS spinner animation with "Analyzing question..." text
- Confirm state: parsed fields grid (question, countries with flags, horizon, category) + Edit/Confirm buttons
- Edit reverts to input state with original question pre-filled
- Confirm calls `forecastClient.confirmSubmission()`, dispatches `submission-confirmed` CustomEvent on window
- Error handling: inline error display on both parse and confirm failures

### SubmissionQueue (`frontend/src/components/SubmissionQueue.ts`)
- Scrollable queue with header showing title and count
- Status-based card rendering:
  - **Pending**: neutral badge, relative time, question, metadata (countries, horizon, category)
  - **Confirmed**: accent badge (waiting for worker pickup)
  - **Processing**: amber badge with CSS pulse animation, elapsed time counter (M:SS format via setInterval)
  - **Complete**: green badge, expandable forecast card via `buildExpandableCard()` from shared utility
  - **Failed**: red badge, error message display
- Empty state: helpful message encouraging submission
- Forecast cache: `Map<string, ForecastResponse>` prevents re-fetching completed forecasts
- Listens for `submission-confirmed` event to auto-refresh
- Proper cleanup: clears all elapsed timers and event listeners on destroy

### Forecasts Screen (`frontend/src/screens/forecasts-screen.ts`)
- Two-column flex layout: 45% left (form), 55% right (queue)
- Module-scoped state matching dashboard/globe pattern
- ScenarioExplorer modal wired for "View Full Analysis" on completed forecasts
- Initial queue load on mount
- RefreshScheduler: 15-second queue polling interval
- Clean unmount: scheduler -> modal -> queue -> form destruction order

### CSS (appended to `frontend/src/styles/panels.css`)
- Forecasts screen layout (`.forecasts-screen`, `.forecasts-col-left`, `.forecasts-col-right`)
- Submission form styles (textarea, footer, submit button, disabled states)
- Parsing spinner (`@keyframes spin`)
- Confirm state grid (field labels, values, action buttons)
- Error display
- Queue header, list, card base styles
- Status badges: pending (neutral), confirmed (accent), processing (amber + `@keyframes pulse-badge`), complete (green), failed (red)
- Elapsed time counter styling with `font-variant-numeric: tabular-nums`
- Empty state
- Complete forecast card override (removes border/bg when nested in queue card)

## Requirements Satisfied

- **SCREEN-04**: Forecasts screen at `/forecasts` with question submission form and processing queue

## Verification

- `npx tsc --noEmit`: zero errors
- `npx vite build`: success (4.43s, all chunks generated)
- No regressions on dashboard or globe screens

## Duration

~5min (Task 1 committed in previous session, Task 2 committed on resume)
