/* Cilia Quest service worker — precache the shell, network-first for the API. */
const CACHE = 'cilia-quest-v1';
const SHELL = ['/', '/index.html', '/style.css', '/app.js',
  '/manifest.webmanifest', '/icon-192.png', '/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then(ks =>
    Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.pathname.startsWith('/api/')) {
    // Always hit the network for data; never serve stale labels/queue.
    e.respondWith(fetch(e.request).catch(() => new Response('{"offline":true}',
      { headers: { 'Content-Type': 'application/json' } })));
    return;
  }
  // Static shell: cache-first.
  e.respondWith(caches.match(e.request).then(r => r || fetch(e.request)));
});
