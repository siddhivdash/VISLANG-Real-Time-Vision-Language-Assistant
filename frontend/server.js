// ============================================================
// SIMPLE HTTP SERVER - Serves files from proper folders
// Run: node server.js
// ============================================================

const http = require('http');
const path = require('path');
const fs = require('fs');
const url = require('url');

const PORT = 3000;

const server = http.createServer((req, res) => {
    // Parse URL
    const parsedUrl = url.parse(req.url, true);
    let pathname = parsedUrl.pathname;

    // Map routes to files
    let filePath;
    
    if (pathname === '/' || pathname === '') {
        filePath = path.join(__dirname, 'public', 'index.html');
    } else if (pathname === '/index.html') {
        filePath = path.join(__dirname, 'public', 'index.html');
    } else if (pathname.startsWith('/js/')) {
        filePath = path.join(__dirname, 'js', pathname.replace('/js/', ''));
    } else if (pathname.startsWith('/css/')) {
        filePath = path.join(__dirname, 'css', pathname.replace('/css/', ''));
    } else {
        filePath = path.join(__dirname, 'public', pathname);
    }

    // Get file extension
    const extname = String(path.extname(filePath)).toLowerCase();
    const mimeTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.wav': 'audio/wav',
        '.mp4': 'video/mp4',
        '.woff': 'application/font-woff',
        '.ttf': 'application/font-ttf',
        '.eot': 'application/vnd.ms-fontobject',
        '.otf': 'application/font-otf',
        '.wasm': 'application/wasm'
    };

    const contentType = mimeTypes[extname] || 'application/octet-stream';

    // Read file
    fs.readFile(filePath, (err, data) => {
        if (err) {
            if (err.code === 'ENOENT') {
                // File not found
                res.writeHead(404, { 'Content-Type': 'text/html' });
                res.end(`<h1>404 - File Not Found</h1><p>Looking for: ${filePath}</p>`, 'utf-8');
            } else {
                // Server error
                res.writeHead(500, { 'Content-Type': 'text/html' });
                res.end(`<h1>500 - Server Error</h1><p>${err}</p>`, 'utf-8');
            }
        } else {
            // File found, send it
            res.writeHead(200, { 'Content-Type': contentType });
            res.end(data, 'utf-8');
        }
    });
});

server.listen(PORT, () => {
    console.log(`✅ Server running on http://localhost:${PORT}`);
    console.log(`📁 Serving files from:
    - public/ → /
    - js/ → /js/
    - css/ → /css/`);
});