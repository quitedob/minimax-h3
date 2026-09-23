#!/usr/bin/env node
// Isolated browser regression checks. No requests are sent to a generation service.
// Usage: node comfyui_download/test_video_ui.cjs [--baseline | --public-smoke]
// --baseline performs GET-only observation of the public UI; normal runs are local mocks.
// --public-smoke explicitly permits ONE real AI preview request, never video generation.
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require('C:/Users/SHUAIBI/AppData/Roaming/npm/node_modules/playwright');

const ROOT = path.resolve(__dirname, '..');
const CHROME = process.env.H3_TEST_CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const evidenceDir = path.join(ROOT, 'docs', 'evidence', `video-ux-20260911-${stamp}`);
fs.mkdirSync(evidenceDir, { recursive: true });
const results = [];
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function artifact(name, data) {
  const destination = path.join(evidenceDir, name);
  fs.writeFileSync(destination, data, { flag: 'wx' });
  return destination;
}

async function baseline(browser) {
  for (const [name, viewport, mobile] of [
    ['mobile-390-before', { width: 390, height: 844 }, true],
    ['desktop-1440-before', { width: 1440, height: 900 }, false],
  ]) {
    const context = await browser.newContext({ viewport, isMobile: mobile, hasTouch: mobile,
      deviceScaleFactor: 1, serviceWorkers: 'block' });
    const allowed = new Set(['/video', '/queue', '/h3/files', '/view', '/favicon.ico']);
    await context.route('**/*', route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin === 'https://video.geekq.xyz:552' && request.method() === 'GET' && allowed.has(url.pathname)) {
        return route.continue();
      }
      return route.abort('blockedbyclient');
    });
    if (context.routeWebSocket) await context.routeWebSocket('**/*', ws => ws.close());
    const page = await context.newPage();
    const errors = [];
    const requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('response', response => requests.push({ url: response.url(), status: response.status() }));
    await page.goto('https://video.geekq.xyz:552/video', { waitUntil: 'domcontentloaded', timeout: 25000 });
    await page.locator('#go').waitFor();
    await page.waitForFunction(() => document.querySelector('#qhead')?.textContent?.trim(), null, { timeout: 15000 }).catch(() => {});
    if (page.locator('body').ariaSnapshot) artifact(`${name}.aria.txt`, await page.locator('body').ariaSnapshot());
    const geometry = await page.evaluate(() => ({
      viewport: { width: innerWidth, height: innerHeight },
      documentWidth: document.documentElement.scrollWidth,
      queue: document.querySelector('#qhead')?.textContent,
      title: document.title,
    }));
    const screenshotPath = path.join(evidenceDir, `${name}.png`);
    await page.screenshot({ path: screenshotPath, fullPage: true });
    artifact(`${name}.json`, JSON.stringify({ geometry, errors, requests }, null, 2));
    console.log(`BASELINE ${screenshotPath}`);
    await context.close();
  }
}

async function main() {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true });
  try {
    if (process.argv.includes('--baseline')) await baseline(browser);
    else if (process.argv.includes('--public-smoke')) await publicSmoke(browser);
    else await regressions(browser);
  } finally {
    await browser.close();
    artifact('summary.json', JSON.stringify(results, null, 2));
    console.log(`EVIDENCE ${evidenceDir}`);
  }
}

async function publicSmoke(browser) {
  const origin = 'https://video.geekq.xyz:552';
  const observation = { aiRequests: [], requests: [], pageErrors: [], blockedWrites: [], viewports: [], media: null };
  for (const [name, viewport, mobile] of [
    ['public-mobile-390', { width: 390, height: 844 }, true],
    ['public-desktop-1440', { width: 1440, height: 900 }, false],
  ]) {
    const context = await browser.newContext({ viewport, isMobile: mobile, hasTouch: mobile,
      deviceScaleFactor: 1, serviceWorkers: 'block' });
    const allowedReadPaths = new Set(['/video', '/queue', '/h3/files', '/view', '/favicon.ico']);
    await context.route('**/*', route => {
      const request = route.request();
      const url = new URL(request.url());
      if (url.origin === origin && request.method() === 'GET' && allowedReadPaths.has(url.pathname)) return route.continue();
      if (url.origin === origin && request.method() === 'HEAD' && url.pathname === '/h3/download') return route.continue();
      if (url.origin === origin && url.pathname === '/h3/prompt' && request.method() === 'POST' &&
          observation.aiRequests.length === 0 && mobile) {
        const body = request.postDataJSON();
        observation.aiRequests.push({ body, startedAt: Date.now() });
        return route.continue();
      }
      if (request.method() !== 'GET') observation.blockedWrites.push({ path: url.pathname, method: request.method() });
      return route.abort('blockedbyclient');
    });
    if (context.routeWebSocket) await context.routeWebSocket('**/*', ws => ws.close());
    const page = await context.newPage();
    page.setDefaultTimeout(15000);
    page.on('pageerror', error => observation.pageErrors.push(error.message));
    page.on('response', response => observation.requests.push({ method: response.request().method(),
      path: new URL(response.url()).pathname, status: response.status() }));
    try {
      await page.goto(`${origin}/video`, { waitUntil: 'domcontentloaded', timeout: 25000 });
      await page.locator('#fileslist .qitem').first().waitFor();
      if (page.locator('body').ariaSnapshot) artifact(`${name}.aria.txt`, await page.locator('body').ariaSnapshot());
      observation.viewports.push(await page.evaluate(() => ({ innerWidth, innerHeight,
        documentWidth: document.documentElement.scrollWidth, title: document.title })));
      await page.screenshot({ path: path.join(evidenceDir, `${name}.png`), fullPage: true });
      if (mobile) {
        const firstCard = page.locator('#fileslist .qitem').first();
        await firstCard.locator('.qprev').click();
        const video = firstCard.locator('video');
        await video.waitFor({ state: 'visible' });
        await until(() => video.evaluate(element => Number.isFinite(element.duration) && element.duration > 0),
          'Public video metadata did not load', 20000);
        observation.media = { durationSeconds: await video.evaluate(element => element.duration),
          source: await video.getAttribute('src'), download: await firstCard.locator('.qdl').evaluate(async element => {
            const response = await fetch(element.href, { method: 'HEAD' });
            return { status: response.status, disposition: response.headers.get('Content-Disposition'),
              contentType: response.headers.get('Content-Type') };
          }) };
        assert.equal(observation.media.download.status, 200);
        assert.match(observation.media.download.disposition || '', /attachment/i);
        await firstCard.locator('.qprev').click();
        await selectSeconds(page, 8);
        await page.locator('#prompt').fill('一只小狗在草地上抬头看镜头，轻微摇尾巴，没有对白。');
        await page.locator('#previewPrompt').check({ force: true });
        const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/h3/prompt', { timeout: 45000 });
        await page.locator('#go').click();
        const response = await responsePromise;
        const requestInfo = observation.aiRequests[0];
        requestInfo.elapsedMs = Date.now() - requestInfo.startedAt;
        requestInfo.status = response.status();
        requestInfo.response = await response.json();
        assert.equal(response.status(), 200, 'Real AI preview request failed');
        assert.match(requestInfo.body.user_text, /这是一个\s*8\s*秒的视频/);
        await page.locator('#finalPrompt').waitFor({ state: 'visible' });
        assert.ok((await page.locator('#finalPrompt').inputValue()).trim().length > 0);
        assert.equal(await page.locator('#confirm').isEnabled(), true);
        // Deliberately DO NOT click #confirm. The generation route is blocked regardless.
        await page.screenshot({ path: path.join(evidenceDir, `${name}-ai-preview.png`), fullPage: true });
      }
    } finally {
      await context.close();
      artifact(`${name}-network.json`, JSON.stringify(observation, null, 2));
    }
  }
  assert.equal(observation.aiRequests.length, 1);
  assert.equal(observation.blockedWrites.length, 0, 'The UI unexpectedly attempted another write');
  assert.equal(observation.pageErrors.length, 0, 'Unhandled page errors during public smoke test');
  assert.equal(observation.requests.filter(request => request.path === '/prompt').length, 0);
  results.push({ name: 'Public desktop/mobile GET smoke plus exactly one eight-second AI preview', pass: true,
    aiElapsedMs: observation.aiRequests[0].elapsedMs, generationRequests: 0 });
  console.log(`PASS PUBLIC: one AI preview HTTP 200 in ${observation.aiRequests[0].elapsedMs}ms; no generation requests`);
}

const fixturePrompt = '<d>[English] <breath> A calm eight-second greeting. </d>\n' +
  '<img data-injected="true" src="x" onerror="window.__injected=true">';
const fixtureTime = Date.UTC(2026, 8, 11, 4, 0, 0);
const fixtureFiles = [
  { filename: 'ux_fixture_00001_.mp4', subfolder: 'video', type: 'output', size: 1597270,
    mtime_ms: fixtureTime, meta: { prompt: fixturePrompt, sec: 8, dur: 123, res: '864×480', steps: 8 } },
  { filename: 'ux_legacy_00001_.mp4', subfolder: 'video', type: 'output', size: 12345,
    mtime_ms: fixtureTime - 1000, meta: {} },
];

async function withFixture(browser, origin, options, run) {
  const mobile = options.mobile !== false;
  const context = await browser.newContext({ viewport: options.viewport || { width: 390, height: 844 },
    isMobile: mobile, hasTouch: mobile, deviceScaleFactor: 1, serviceWorkers: 'block' });
  const state = { requests: [], errors: [], blocked: [], promptCalls: [], aiCalls: [], metaCalls: [],
    queueCalls: 0, fileCalls: 0, failQueue: false, failFiles: false, failMeta: false, failAI: false,
    histories: {}, queue: { queue_running: [], queue_pending: [] }, files: fixtureFiles, ...options.state };
  if (options.storage) await context.addInitScript(values => {
    for (const [key, value] of Object.entries(values)) localStorage.setItem(key, JSON.stringify(value));
  }, options.storage);
  const videoPath = path.join(ROOT, 'ComfyUI_sage3_py312', 'output', 'video',
    'web_bench_h3_turbo_v4_8step_solattn_26d816e_00001_.mp4');
  const videoBytes = fs.existsSync(videoPath) ? fs.readFileSync(videoPath) : null;
  const json = (route, data, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(data) });
  await context.route('**/*', async route => {
    const request = route.request();
    const url = new URL(request.url());
    if (url.origin !== origin) {
      state.blocked.push(request.url());
      return route.abort('blockedbyclient');
    }
    const pathname = url.pathname;
    let body = null;
    try { body = request.postDataJSON(); } catch (_) { /* Multipart image uploads are mocked too. */ }
    state.requests.push({ path: pathname, method: request.method(), body });
    if (pathname === '/video' && request.method() === 'GET') return route.continue();
    if (pathname === '/queue') {
      state.queueCalls++;
      return json(route, state.failQueue ? { error: 'Fixture queue connection unavailable' } :
        state.queue, state.failQueue ? 503 : 200);
    }
    if (pathname === '/h3/files') {
      state.fileCalls++;
      return json(route, state.failFiles ? { error: 'Fixture gallery temporarily unavailable' } : state.files,
        state.failFiles ? 503 : 200);
    }
    if (pathname === '/h3/prompt') {
      state.aiCalls.push(body);
      await sleep(60);
      if (state.failAI) return json(route, { error: 'AI 暂时不可用，请稍后重试。' }, 500);
      return json(route, { h3_prompt: 'A single continuous shot for eight seconds. <d>[English] Hello.</d>',
        image_description: '<img data-injected="true" src="x" onerror="window.__injected=true"> A peaceful lake.' });
    }
    if (pathname === '/upload/image') return json(route, { name: 'ux-reference.png', subfolder: '', type: 'input' });
    if (pathname === '/prompt') {
      state.promptCalls.push(body);
      const pid = `ux-prompt-${state.promptCalls.length}`;
      const graph = body.prompt;
      const saveNode = Object.values(graph).find(node => node.class_type === 'SaveVideo');
      const filename = (saveNode?.inputs?.filename_prefix || 'video/web_fixture').split('/').pop() + '_00001_.mp4';
      const now = Date.now();
      state.histories[pid] = { prompt: [1, pid, graph, { create_time: now - 14000 }, []],
        status: { status_str: 'success', completed: true,
          messages: [['execution_start', { timestamp: now - 12000 }], ['execution_success', { timestamp: now }]] },
        outputs: { save: { images: [{ filename, subfolder: 'video', type: 'output' }] } } };
      return json(route, { prompt_id: pid, number: state.promptCalls.length, node_errors: {} });
    }
    if (pathname.startsWith('/history/')) {
      const pid = decodeURIComponent(pathname.slice('/history/'.length));
      return json(route, state.histories[pid] ? { [pid]: state.histories[pid] } : {});
    }
    if (pathname === '/h3/meta') {
      state.metaCalls.push(body);
      return json(route, state.failMeta ? { error: 'Fixture metadata write temporarily unavailable' } : { ok: true }, state.failMeta ? 503 : 200);
    }
    if (pathname === '/view' || pathname === '/h3/download') {
      if (!videoBytes) return route.fulfill({ status: 404, body: 'Optional local video fixture not present' });
      return route.fulfill({ status: 200, contentType: 'video/mp4', body: videoBytes,
        headers: pathname === '/h3/download' ? { 'Content-Disposition': 'attachment; filename="ux-fixture.mp4"' } : {} });
    }
    if (pathname === '/favicon.ico') return route.fulfill({ status: 204, body: '' });
    state.blocked.push(request.url());
    return route.abort('blockedbyclient');
  });
  if (context.routeWebSocket) await context.routeWebSocket('**/*', ws => ws.close());
  else await context.addInitScript(() => { window.WebSocket = class { close() {} }; });
  const page = await context.newPage();
  page.setDefaultTimeout(7000);
  page.on('pageerror', error => state.errors.push(error.message));
  try {
    await page.goto(`${origin}/video`, { waitUntil: 'domcontentloaded' });
    await page.locator('#go').waitFor();
    await run(page, state, Boolean(videoBytes));
    assert.equal(state.blocked.length, 0, `Unexpected resource requests: ${state.blocked.join(', ')}`);
  } catch (error) {
    const label = String(options.name || 'fixture').replace(/[^a-z0-9]+/gi, '-').slice(0, 70);
    artifact(`${label}-failure.json`, JSON.stringify({ error: error.message, state,
      ui: await page.evaluate(() => ({ status: document.querySelector('#status')?.textContent,
        error: document.querySelector('#err')?.textContent,
        go: { hidden: document.querySelector('#go')?.hidden, disabled: document.querySelector('#go')?.disabled },
        confirm: { hidden: document.querySelector('#confirm')?.hidden, disabled: document.querySelector('#confirm')?.disabled },
        previewClass: document.querySelector('#prompt-preview')?.className,
        resultClass: document.querySelector('#result')?.className,
      })) }, null, 2));
    await page.screenshot({ path: path.join(evidenceDir, `${label}-failure.png`), fullPage: true });
    throw error;
  } finally {
    await context.close();
  }
}

async function until(predicate, message, timeout = 12000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await sleep(100);
  }
  assert.fail(message);
}

async function selectSeconds(page, seconds) {
  const locator = page.locator('#seconds');
  const tag = await locator.evaluate(element => element.tagName);
  if (tag === 'SELECT') await locator.selectOption(String(seconds));
  else await locator.fill(String(seconds));
}

async function generated(page, state) {
  await until(() => state.promptCalls.length > 0, 'The fixture never received a generation request');
  await until(() => page.locator('#go').isEnabled(), 'Completed generation did not unlock the main button');
  await page.locator('#result.visible').waitFor();
  assert.equal(state.errors.length, 0, `Unhandled errors: ${state.errors.join('; ')}`);
}

async function regressions(browser) {
  const sourcePath = path.join(ROOT, 'comfyui_download', 'h3_web_queue', 'web', 'index.html');
  const workflowPath = path.join(ROOT, 'comfyui_download', 'cloud_h3_sage3_solattn_easycache_prompt.json');
  const server = http.createServer((request, response) => {
    // There is deliberately no proxy or write handler in this server.
    if (request.method !== 'GET' || request.url?.split('?')[0] !== '/video') {
      response.writeHead(404); response.end('No fixture route'); return;
    }
    const workflow = JSON.parse(fs.readFileSync(workflowPath, 'utf8').replace(/^\uFEFF/, ''));
    const html = fs.readFileSync(sourcePath, 'utf8').replace('__H3_WORKFLOW_JSON__',
      JSON.stringify(workflow).replace(/</g, '\\u003c'));
    response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cache-Control': 'no-store' });
    response.end(html);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const origin = `http://127.0.0.1:${server.address().port}`;
  async function test(name, options, run) {
    const filter = process.argv.find(argument => argument.startsWith('--case='))?.slice(7);
    if (filter && !name.toLowerCase().includes(filter.toLowerCase())) return;
    const start = Date.now();
    try {
      await withFixture(browser, origin, { ...options, name }, run);
      results.push({ name, pass: true, elapsedMs: Date.now() - start });
      console.log(`PASS ${name}`);
    } catch (error) {
      results.push({ name, pass: false, elapsedMs: Date.now() - start, error: error.message });
      console.error(`FAIL ${name}: ${error.message}`);
    }
  }
  try {
    await test('Eight-second AI preview, editable final prompt, single submission, success unlock', {}, async (page, state) => {
      await selectSeconds(page, 8);
      await page.locator('#prompt').fill('A puppy rests in the grass.');
      await page.locator('#go').click();
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      assert.equal(state.aiCalls.length, 1);
      assert.match(state.aiCalls[0].user_text, /这是一个\s*8\s*秒的视频/);
      assert.equal(await page.locator('#prompt').inputValue(), 'A puppy rests in the grass.');
      assert.equal(await page.locator('#seconds').isDisabled(), true, 'Duration must stay consistent while awaiting confirmation');
      const edited = 'Edited final prompt. <d>[English] <breath> Hello again. </d>';
      await page.locator('#finalPrompt').fill(edited);
      await page.locator('#confirm').click({ clickCount: 2, delay: 20 });
      await generated(page, state);
      assert.equal(state.promptCalls.length, 1, 'Double click submitted duplicate expensive work');
      const graph = state.promptCalls[0].prompt;
      assert.equal(graph.cond.inputs.prompt, edited);
      assert.equal(graph.cond.inputs.length, 192);
      assert.equal(graph.sigmas.inputs.steps, 8);
    });

    await test('AI direct-submit path includes duration without an in-graph AI bypass', {}, async (page, state) => {
      await selectSeconds(page, 8);
      await page.locator('#prompt').fill('A gentle camera move across a garden.');
      await page.locator('#previewPrompt').uncheck({ force: true });
      await page.locator('#go').click();
      await generated(page, state);
      assert.equal(state.aiCalls.length, 1);
      assert.match(state.aiCalls[0].user_text, /这是一个\s*8\s*秒的视频/);
      assert.equal(typeof state.promptCalls[0].prompt.cond.inputs.prompt, 'string');
      assert.equal(state.promptCalls[0].prompt.h3ds, undefined);
    });

    await test('AI off leaves original prompt and acting tags unchanged', {}, async (page, state) => {
      const original = 'One quiet shot. <d>[English] <breath> Hello. <pause> </d>';
      await page.locator('#prompt').fill(original);
      await page.locator('#deepseek').uncheck({ force: true });
      await page.locator('#previewPrompt').uncheck({ force: true });
      await page.locator('#go').click();
      await generated(page, state);
      assert.equal(state.aiCalls.length, 0);
      assert.equal(state.promptCalls[0].prompt.cond.inputs.prompt, original);
    });

    await test('Reference-image AI receives eight seconds and treats model descriptions as text', {}, async (page, state) => {
      await page.locator('[data-pane="img"]').click();
      await selectSeconds(page, 8);
      await page.locator('#prompt-img').fill('Keep the lake peaceful.');
      await page.locator('#refimg').setInputFiles({ name: 'reference.png', mimeType: 'image/png',
        buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOt0AAAAASUVORK5CYII=', 'base64') });
      await page.locator('#go').click();
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      assert.equal(state.aiCalls.length, 1);
      assert.match(state.aiCalls[0].user_text, /这是一个\s*8\s*秒的视频/);
      assert.equal(state.aiCalls[0].mode, 'Ref2VA');
      assert.equal(state.aiCalls[0].image_name, 'ux-reference.png');
      assert.equal(await page.locator('[data-injected]').count(), 0);
      assert.equal(await page.evaluate(() => Boolean(window.__injected)), false);
    });

    for (const useAI of [true, false]) {
      for (const previewFirst of [true, false]) {
        await test(`Image workflow AI=${useAI} preview=${previewFirst} preserves prompt and duration`, {}, async (page, state) => {
          await page.locator('[data-pane="img"]').click();
          await selectSeconds(page, 8);
          const original = '<Picture 1> A quiet lake. <d>[English] Hello.</d>';
          await page.locator('#prompt-img').fill(original);
          await page.locator('#refimg').setInputFiles({ name: 'reference.png', mimeType: 'image/png',
            buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aOt0AAAAASUVORK5CYII=', 'base64') });
          if (!useAI) await page.locator('#aidImg').uncheck({ force: true });
          if (!previewFirst) await page.locator('#previewPrompt').uncheck({ force: true });
          await page.locator('#go').click();
          let expectedPrompt = original;
          if (previewFirst) {
            await page.locator('#finalPrompt').waitFor({ state: 'visible' });
            expectedPrompt = 'Edited image prompt with <Picture 1>. <d>[English] Hello.</d>';
            await page.locator('#finalPrompt').fill(expectedPrompt);
            await page.locator('#confirm').click();
          }
          await generated(page, state);
          assert.equal(state.aiCalls.length, useAI ? 1 : 0);
          if (useAI) assert.match(state.aiCalls[0].user_text, /这是一个\s*8\s*秒的视频/);
          const graph = state.promptCalls[0].prompt;
          assert.equal(graph.cond.class_type, 'MiniMaxH3ReferenceToVideo');
          assert.equal(graph.cond.inputs.length, 192);
          if (previewFirst || !useAI) assert.equal(graph.cond.inputs.prompt, expectedPrompt);
        });
      }
    }

    await test('Reload restores editable pending draft without repeating AI or submitting', {}, async (page, state) => {
      await selectSeconds(page, 8);
      await page.locator('#prompt').fill('A preserved draft idea.');
      await page.locator('#go').click();
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      await page.locator('#finalPrompt').fill('A carefully edited pending draft.');
      await page.reload({ waitUntil: 'domcontentloaded' });
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      assert.equal(await page.locator('#finalPrompt').inputValue(), 'A carefully edited pending draft.');
      assert.equal(await page.locator('#prompt').inputValue(), 'A preserved draft idea.');
      assert.equal(await page.locator('#seconds').inputValue(), '8');
      assert.equal(state.aiCalls.length, 1);
      assert.equal(state.promptCalls.length, 0);
      assert.equal(await page.locator('#confirm').isEnabled(), true);
    });

    await test('Keyboard switches tabs and reaches image upload without a mouse', { mobile: false }, async page => {
      await page.locator('#tab-text').focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await page.locator('#tab-img').getAttribute('aria-selected'), 'true');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'tab-img');
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'refimg');
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement?.id), 'prompt-img');
    });

    await test('Empty idea cannot invoke AI or submit work', {}, async (page, state) => {
      assert.equal(await page.locator('#go').isDisabled(), true);
      await page.locator('#prompt').fill('   ');
      assert.equal(await page.locator('#go').isDisabled(), true);
      assert.equal(state.aiCalls.length, 0);
      assert.equal(state.promptCalls.length, 0);
    });

    await test('AI failure is recoverable and never silently submits the original idea', { state: { failAI: true } }, async (page, state) => {
      await page.locator('#prompt').fill('An idea whose AI service is unavailable.');
      await page.locator('#go').click();
      await page.locator('#err').waitFor({ state: 'visible' });
      assert.match(await page.locator('#err').innerText(), /稍后|重试/);
      assert.equal(await page.locator('#go').isEnabled(), true);
      assert.equal(state.aiCalls.length, 1);
      assert.equal(state.promptCalls.length, 0);
    });

    await test('Back from preview preserves final editing and unlocks video settings', {}, async page => {
      await page.locator('#prompt').fill('An original idea.');
      await page.locator('#go').click();
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      await page.locator('#finalPrompt').fill('My carefully revised prompt.');
      await page.locator('#back').click();
      assert.equal(await page.locator('#seconds').isDisabled(), false);
      assert.equal(await page.locator('#finalPrompt').inputValue(), 'My carefully revised prompt.');
    });

    await test('Empty final prompt cannot submit a generation task', {}, async (page, state) => {
      await page.locator('#prompt').fill('A short scene for validation.');
      await page.locator('#deepseek').uncheck({ force: true });
      await page.locator('#go').click();
      await page.locator('#finalPrompt').waitFor({ state: 'visible' });
      await page.locator('#finalPrompt').fill('   ');
      if (await page.locator('#confirm').isEnabled()) await page.locator('#confirm').click();
      await sleep(200);
      assert.equal(state.promptCalls.length, 0);
      assert.equal(await page.locator('#finalPrompt').isVisible(), true);
    });

    for (const [name, viewport, mobile] of [
      ['mobile-320', { width: 320, height: 740 }, true],
      ['mobile-390', { width: 390, height: 844 }, true],
      ['desktop-1440', { width: 1440, height: 900 }, false],
    ]) {
      await test(`${name}: no overflow, visible media metadata, literal prompt tags`, { viewport, mobile }, async page => {
        await page.locator('#fileslist .qitem').first().waitFor();
        const geometry = await page.evaluate(() => ({ width: innerWidth, documentWidth: document.documentElement.scrollWidth,
          promptFont: parseFloat(getComputedStyle(document.querySelector('#prompt')).fontSize) }));
        assert.ok(geometry.documentWidth <= viewport.width + 1, `Horizontal overflow: ${JSON.stringify(geometry)}`);
        if (mobile) assert.ok(geometry.promptFont >= 16, 'Mobile input font is below 16px');
        const galleryText = await page.locator('#fileslist').textContent();
        assert.ok(galleryText.includes('<d>[English]'), 'Acting tags must remain visible text');
        assert.match(galleryText, /视频时长/);
        assert.match(galleryText, /生成用时|生成时长/);
        assert.match(galleryText, /未记录/);
        assert.equal(await page.locator('[data-injected]').count(), 0);
        assert.equal(await page.evaluate(() => Boolean(window.__injected)), false);
        await page.screenshot({ path: path.join(evidenceDir, `${name}-after.png`), fullPage: true });
      });
    }

    await test('Inline video and gallery DOM survive periodic refresh', {}, async (page, state, hasVideo) => {
      await page.locator('#fileslist .qitem').first().waitFor();
      const firstCard = await page.locator('#fileslist .qitem').first().elementHandle();
      await page.locator('#fileslist .qprev').first().click();
      const video = page.locator('#fileslist video').first();
      await video.waitFor({ state: 'visible' });
      const videoNode = await video.elementHandle();
      if (hasVideo) {
        await video.evaluate(async element => { element.muted = true; element.loop = true; await element.play(); });
      }
      const pollsBefore = state.fileCalls;
      await until(() => state.fileCalls > pollsBefore, 'Gallery refresh did not occur');
      await sleep(300);
      assert.equal(await firstCard.evaluate(element => element.isConnected), true, 'Unchanged gallery card was rebuilt');
      assert.equal(await videoNode.evaluate(element => element.isConnected), true, 'Video element was removed during refresh');
      if (hasVideo) assert.equal(await video.evaluate(element => element.paused), false, 'Refresh paused video playback');
    });

    const longTaskGraph = { cond: { inputs: { prompt: 'A long and detailed description of a peaceful landscape. '.repeat(12),
      length: 192, width: 864, height: 480 } }, sigmas: { class_type: 'BasicScheduler', inputs: { steps: 8 } } };
    const runningRecord = [1, 'ux-running', longTaskGraph, { create_time: fixtureTime }, []];
    const pendingRecord = [2, 'ux-pending', longTaskGraph, { create_time: fixtureTime }, []];
    await test('Expanded running and queued prompt details survive the five-second refresh', {
      state: { queue: { queue_running: [runningRecord], queue_pending: [pendingRecord] } },
    }, async (page, state) => {
      await page.locator('#tlist .qitem').first().waitFor();
      const nodes = [];
      for (const pid of ['ux-running', 'ux-pending']) {
        const details = page.locator(`#tlist [data-pid="${pid}"] .prompt-full`);
        await details.locator('summary').click();
        nodes.push(await details.elementHandle());
      }
      const pollsBefore = state.queueCalls;
      await until(() => state.queueCalls > pollsBefore, 'Task refresh never occurred');
      await sleep(200);
      for (const node of nodes) assert.equal(await node.evaluate(element => element.isConnected && element.open), true);
    });

    const sixFiles = Array.from({ length: 6 }, (_, index) => ({ ...fixtureFiles[0],
      filename: `ux_initial_${index + 1}.mp4`, mtime_ms: fixtureTime - index * 1000 }));
    await test('A newly prepended video does not remove the sixth visible playing card', {
      state: { files: sixFiles },
    }, async (page, state, hasVideo) => {
      await page.locator('#fileslist .qitem').nth(5).waitFor();
      const sixthCard = page.locator('#fileslist .qitem').nth(5);
      const cardNode = await sixthCard.elementHandle();
      await sixthCard.locator('.qprev').click();
      const video = sixthCard.locator('video');
      const videoNode = await video.elementHandle();
      if (hasVideo) await video.evaluate(async element => { element.muted = true; element.loop = true; await element.play(); });
      state.files = [{ ...fixtureFiles[0], filename: 'ux_new_shared_result.mp4', mtime_ms: fixtureTime + 1000 }, ...sixFiles];
      await page.locator('#refresh').click();
      await until(() => page.locator('#fileslist .qitem').count().then(count => count === 7), 'Previously visible sixth card was dropped');
      assert.equal(await cardNode.evaluate(element => element.isConnected), true);
      assert.equal(await videoNode.evaluate(element => element.isConnected), true);
      if (hasVideo) assert.equal(await videoNode.evaluate(element => element.paused), false);
    });

    await test('Restored failed history with completed=false clears stale pending metadata', {
      storage: { h3pending: { 'ux-failed-old': { jobId: 'old-fixture', prompt: 'An old failed request', sec: 8, steps: 8, res: '864×480' } } },
      state: { histories: { 'ux-failed-old': { status: { status_str: 'error', completed: false,
        messages: [['execution_error', { timestamp: fixtureTime, message: 'Fixture interrupted' }]] }, outputs: {} } } },
    }, async (page, state) => {
      await until(() => page.evaluate(() => Object.keys(JSON.parse(localStorage.getItem('h3pending') || '{}')).length === 0),
        'The old failed job remains pending forever because completed=false');
      assert.equal(state.metaCalls.length, 0);
      assert.equal(state.promptCalls.length, 0);
    });

    await test('Failed metadata remains pending and retries successfully', { state: { failMeta: true } }, async (page, state) => {
      await page.locator('#prompt').fill('A metadata retry fixture.');
      await page.locator('#deepseek').uncheck({ force: true });
      await page.locator('#previewPrompt').uncheck({ force: true });
      await page.locator('#go').click();
      await generated(page, state);
      await until(() => state.metaCalls.length > 0, 'Completed job metadata was never sent');
      assert.ok(await page.evaluate(() => Object.keys(JSON.parse(localStorage.getItem('h3pending') || '{}')).length > 0),
        'Failed metadata write incorrectly discarded pending record');
      state.failMeta = false;
      await until(() => state.metaCalls.length > 1, 'Metadata was never retried');
      await until(() => page.evaluate(() => Object.keys(JSON.parse(localStorage.getItem('h3pending') || '{}')).length === 0),
        'Successful metadata retry did not clear pending state');
    });

    await test('Queue or gallery failure shows a recoverable visible state', { state: { failQueue: true, failFiles: true } }, async page => {
      await until(async () => /无法|失败|连接|重试|稍后/.test(await page.locator('#queue-panel').innerText()),
        'Both API calls failed but the UI showed no useful error');
      const text = await page.locator('#queue-panel').innerText();
      assert.match(text, /重试|稍后|恢复/);
    });
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
  if (results.some(result => !result.pass)) process.exitCode = 1;
}

main().catch(error => { console.error(error.stack || error); process.exitCode = 1; });
