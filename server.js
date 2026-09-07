// server.js
// Backend with TWO endpoints: /chat (text Q&A) and /transcribe (voice-to-text)

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const fs = require('fs');
const multer = require('multer');
const { GoogleGenAI } = require('@google/genai');
const Groq = require('groq-sdk');

const app = express();
app.use(cors());
app.use(express.json());

const upload = multer({ dest: 'uploads/' });

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });
const knowledgeBase = JSON.parse(fs.readFileSync('knowledge-base.json', 'utf-8'));

const SIMILARITY_THRESHOLD = 0.5;
const TOP_K = 3;

function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function embedQuery(query) {
  const result = await ai.models.embedContent({
    model: 'gemini-embedding-001',
    contents: query,
    config: { taskType: 'RETRIEVAL_QUERY' },
  });
  return result.embeddings[0].values;
}

function retrieveTopChunks(queryEmbedding) {
  const scored = knowledgeBase.map((chunk) => ({
    text: chunk.text,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, TOP_K);
}

async function answerQuestion(question) {
  const queryEmbedding = await embedQuery(question);
  const topChunks = retrieveTopChunks(queryEmbedding);
  const goodMatches = topChunks.filter((c) => c.score >= SIMILARITY_THRESHOLD);

  if (goodMatches.length === 0) {
    return "I don't have information about that. Please ask something related to business schemes or financial planning.";
  }

  const context = goodMatches.map((c) => c.text).join('\n\n');
  const systemPrompt = `You are a financial and business advisory assistant for rural micro-entrepreneurs in India.
Answer using ONLY the context below. If it doesn't fully answer the question, say what you can and note the gap.
Never invent scheme details that aren't in the context.

Context:
${context}`;

  const response = await groq.chat.completions.create({
    model: 'openai/gpt-oss-120b',
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: question },
    ],
  });

  return response.choices[0].message.content;
}

app.post('/chat', async (req, res) => {
  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: 'Missing "question"' });
    const answer = await answerQuestion(question);
    res.json({ answer });
  } catch (err) {
    console.error('Error in /chat:', err.message);
    res.status(500).json({ error: 'Something went wrong on the server' });
  }
});

app.post('/transcribe', upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: 'No audio file received' });

    const transcription = await groq.audio.transcriptions.create({
      file: fs.createReadStream(req.file.path),
      model: 'whisper-large-v3-turbo',
      temperature: 0,
      response_format: 'verbose_json',
    });

    fs.unlink(req.file.path, () => {});

    res.json({ text: transcription.text });
  } catch (err) {
    console.error('Error in /transcribe:', err.message);
    res.status(500).json({ error: 'Transcription failed' });
  }
});

app.get('/', (req, res) => res.send('SIH chatbot backend is running.'));

const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', () => console.log(`Server running on port ${PORT}`));