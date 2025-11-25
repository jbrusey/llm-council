export function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return null;
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }
  return `${seconds.toFixed(2)} s`;
}
