let sessionId = null;
let showDetails = false;

const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');
const toggleDetailsBtn = document.getElementById('toggleDetailsBtn');
const sessionChip = document.getElementById('sessionChip');

const sqlBox = document.getElementById('sqlBox');
const factsBox = document.getElementById('factsBox');
const faqBox = document.getElementById('faqBox');

function addMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatEl.appendChild(wrap);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function setDetails(debug) {
  sqlBox.textContent = debug?.executed_sql || '';
  factsBox.textContent = debug?.facts || '';
  faqBox.textContent = debug?.faq_excerpt || '';
}

async function ensureSession() {
  if (sessionId) return;
  const res = await fetch('/api/session', { method: 'POST' });
  const data = await res.json();
  sessionId = data.session_id;
  sessionChip.textContent = 'session: ' + sessionId.slice(0, 8);
}

async function sendMessage() {
  const msg = inputEl.value.trim();
  if (!msg) return;
  inputEl.value = '';
  autoResize();

  await ensureSession();
  addMessage('user', msg);
  addMessage('assistant', '…');
  const lastBubble = chatEl.lastChild.querySelector('.bubble');

  sendBtn.disabled = true;
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: msg })
    });
    const data = await res.json();
    if (data.error) {
      lastBubble.textContent = data.error;
    } else {
      lastBubble.textContent = data.answer || '';
      setDetails(data.debug || {});
    }
  } catch (e) {
    lastBubble.textContent = 'Error: ' + (e?.message || e);
  } finally {
    sendBtn.disabled = false;
  }
}

function autoResize() {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
}

sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('input', autoResize);
inputEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

newChatBtn.addEventListener('click', async () => {
  // start a new session
  sessionId = null;
  sessionChip.textContent = 'session: —';
  chatEl.innerHTML = '';
  sqlBox.textContent = '';
  factsBox.textContent = '';
  faqBox.textContent = '';
  addMessage('assistant', 'Hi! Ask me anything about the model or the FAQ.');
});

const contentEl = document.querySelector('.content');
toggleDetailsBtn.addEventListener('click', () => {
  showDetails = !showDetails;
  contentEl.classList.toggle('show-details', showDetails);
});

// initial greeting
addMessage('assistant', 'Hi! Ask me a question about the company (financial projections, metrics, or investor FAQs).');
