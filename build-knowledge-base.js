// build-knowledge-base.js
// Run this ONCE locally (node build-knowledge-base.js) to create knowledge-base.json.
// Commit the generated knowledge-base.json to your repo — server.js reads it directly.
// You do NOT need to run this on Render.

require('dotenv').config();
const fs = require('fs');
const { GoogleGenAI } = require('@google/genai');

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

const rawChunks = [
  `MUDRA loan scheme (Pradhan Mantri Mudra Yojana / PMMY) offers collateral-free loans to
   non-corporate, non-farm small and micro enterprises. Categories: Shishu (up to ₹50,000),
   Kishor (₹50,000 to ₹5 lakh), Tarun (₹5 lakh to ₹10 lakh), and Tarun Plus (₹10 lakh to
   ₹20 lakh, only for entrepreneurs who previously repaid a Tarun loan). Applicants must be
   Indian citizens aged 18 to 65.`,

  `PM-AJAY (Pradhan Mantri Anusuchit Jaati Abhyuday Yojana) is a centrally sponsored scheme by
   the Ministry of Social Justice and Empowerment for socio-economic upliftment of Scheduled
   Caste communities. Includes grants-in-aid for income-generating projects (subsidy of
   ₹10,000 per beneficiary or 50% of loan, whichever is lower), skill development training,
   and hostel construction.`,

  `Stand-Up India scheme provides composite loans of ₹10 lakh to ₹1 crore to SC, ST, and women
   entrepreneurs for new "greenfield" enterprises. Every bank branch must extend at least one
   loan to an SC/ST borrower and one to a woman borrower. Repayment up to 7 years with up to
   18 months moratorium.`,

  `NSFDC Micro Finance Scheme (MFS): for units costing up to ₹1.40 lakh. Loan up to 90% of
   project cost, max ₹1.25 lakh per unit. Interest rate 6.5% to beneficiaries. Repaid in
   quarterly instalments within 3 years, including a 3-month moratorium.`,

  `NSFDC Term Loan: for units costing more than ₹1.40 lakh up to ₹50 lakh. Loan up to 90% of
   project cost, max ₹45 lakh per unit. Interest rate 8% to beneficiaries. Repaid in quarterly
   instalments within 7 years, including a 6-month moratorium (12 months for plantation and
   construction activities).`,

  `To increase customers for a small local business: encourage word-of-mouth referrals, offer
   small discounts for repeat customers, use WhatsApp Business to share photos of new stock,
   and keep consistent shop timings.`,

  // Add more chunks here as needed.
];

async function build() {
  const knowledgeBase = [];
  for (const text of rawChunks) {
    const result = await ai.models.embedContent({
      model: 'gemini-embedding-001',
      contents: text,
      config: { taskType: 'RETRIEVAL_DOCUMENT' },
    });
    knowledgeBase.push({ text, embedding: result.embeddings[0].values });
    console.log('Embedded:', text.slice(0, 50) + '...');
  }
  fs.writeFileSync('knowledge-base.json', JSON.stringify(knowledgeBase, null, 2));
  console.log(`\nSaved ${knowledgeBase.length} chunks to knowledge-base.json`);
}

build().catch((err) => console.error('Error:', err.message));