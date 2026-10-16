/** Results contain server filesystem paths; demo reports contain inline text. */
export function artifactFilename(reference: string | undefined, fallback: string): string {
  if (!reference || /[\r\n]/.test(reference)) return fallback;
  const filename = reference.replace(/\\/g, '/').split('/').pop()?.trim();
  const suffix = fallback.slice(fallback.lastIndexOf('.'));
  return filename && filename.toLowerCase().endsWith(suffix.toLowerCase()) ? filename : fallback;
}

export function downloadText(content: string, filename: string, type = 'text/plain;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement('a');
  link.href = url; link.download = filename;
  document.body.appendChild(link); link.click(); link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function serializeCsv(rows: Record<string, unknown>[], columns: string[]): string {
  const cell = (value: unknown) => {
    let text = value == null ? '' : String(value);
    if (typeof value === 'string' && /^[=+@\-\t\r]/.test(text)) text = `'${text}`;
    return `"${text.replace(/"/g, '""')}"`;
  };
  return '\uFEFF' + [columns.map(cell).join(','), ...rows.map((row) => columns.map((key) => cell(row[key])).join(','))].join('\r\n');
}
