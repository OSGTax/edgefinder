// Turns the single-file build (dist-single/index.html) into a bare page body
// — <title>, <style>, the app root and the inline script — for hosts that wrap
// pages in their own <html>/<head> skeleton.
import { readFileSync, writeFileSync } from 'node:fs';

const html = readFileSync('dist-single/index.html', 'utf8');
const title = html.match(/<title>[\s\S]*?<\/title>/)?.[0] ?? '<title>Grass Stain League</title>';
const styles = [...html.matchAll(/<style[^>]*>[\s\S]*?<\/style>/g)].map((m) => m[0]).join('\n');
const scripts = [...html.matchAll(/<script[^>]*>[\s\S]*?<\/script>/g)].map((m) => m[0]).join('\n');
const page = `${title}\n${styles}\n<div id="app"></div>\n${scripts}\n`;
writeFileSync('dist-single/page.html', page);
console.log(`dist-single/page.html: ${(page.length / 1024).toFixed(0)} KB`);
