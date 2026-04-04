# Dataset Description — AI Knowledge Helper (RAG System)

---

## Why Pakistani Banking FAQ Dataset?

### The Core Reasoning Behind This Choice

The entire purpose of a **Retrieval-Augmented Generation (RAG)** system is to give an LLM access to **private, domain-specific knowledge** that it was never trained on. If you use a general knowledge dataset — like Wikipedia articles or ML research papers — a standard LLM like GPT-4 or Claude can already answer those questions without any retrieval system. That completely defeats the purpose of building RAG.

Pakistani banking FAQ documents solve this problem perfectly, for the following reasons:

### ✅ Reason 1: LLMs Have Zero Knowledge of This Data
No public LLM has been trained on the internal FAQ policies of HBL, Meezan Bank, Allied Bank, or Bank Alfalah. These are institution-specific documents covering Pakistani regulations, local products, and Shariah-compliant banking terms. A general LLM simply cannot answer:
- *"What is the daily cash withdrawal limit on a Meezan Digital Freelancer Account?"*
- *"What is the late payment charge on an Alfalah Personal Loan?"*
- *"What SWIFT code should I use for Meezan Roshan Digital Account remittances?"*

### ✅ Reason 2: Perfect RAG Use Case — Question-Answer Format
FAQs are already structured as **Question → Answer** pairs. This is exactly the format RAG is designed to handle. The retrieval system can match an incoming user question to the most relevant FAQ chunk and pass it to the LLM for a clean, grounded answer.

### ✅ Reason 3: Rich, Dense Text Content
Banking FAQ documents are text-heavy with precise legal and procedural language. They are ideal for:
- **Chunking** (300–500 token windows)
- **Embedding** (meaningful semantic content)
- **Retrieval** (distinct topics per chunk)

### ✅ Reason 4: Real Business Value
Every Pakistani bank wants a customer support chatbot that can answer user queries from their own policy documents. This project directly demonstrates that use case — making it highly relevant and impressive for an AI internship evaluation.

### ✅ Reason 5: Authentic, Publicly Available Sources
All 19 documents are sourced directly from official bank websites — no scraping, no copyright issues, no fabricated data. Every PDF link is a direct download from the bank's own servers.

### ✅ Reason 6: Diverse Coverage Within One Domain
The dataset covers multiple sub-topics under one unified domain (banking), which is ideal for RAG evaluation:
- Account opening procedures
- Digital/mobile banking
- Loans and financing
- Debit/credit cards
- Remittance and overseas accounts
- Freelancer banking
- Islamic/Shariah-compliant products

---

## 🏦 Bank 1: HBL — Habib Bank Limited

### About HBL
Habib Bank Limited (HBL) is **Pakistan's largest bank**, founded in 1941. With over 1,700 branches, 2,000+ ATMs, and 20 million customers across 15 countries, HBL is the most prominent commercial bank in Pakistan. It offers conventional and Islamic banking products and is headquartered in Karachi.

---

### 📄 HBL Document 1

| Field | Details |
|---|---|
| **Document Name** | HBL Internet Payment Gateway (IPG) FAQs |
| **Source Link** | https://www.hbl.com/assets/documents/HBL_IPG_FAQs.pdf |
| **Document Type** | FAQ / Policy Document |
| **Topic** | Online merchant payment gateway setup and requirements |

**What It Contains:**
- Required documents to apply for HBL Internet Payment Gateway
- Business registration requirements for merchants
- Age and account eligibility criteria (minimum 25 years for individual applicants)
- HBL account requirements (mandatory business account, not personal)
- Geographic coverage including Hunza and Gilgit
- Instructions for account status and outstanding balance conditions
- Step-by-step onboarding checklist for e-commerce businesses

**Why Useful for RAG:**
Merchants and e-commerce businesses frequently ask about payment gateway setup. This document provides very specific procedural answers that a general LLM cannot provide.

---

### 📄 HBL Document 2

| Field | Details |
|---|---|
| **Document Name** | HBL Fast Transfer — Home Remittance FAQs |
| **Source Link** | https://www.hbl.com/assets/documents/FAQS-_HBL_Fast_Transfer.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Home remittance service for overseas Pakistanis |

**What It Contains:**
- Definition of Home Remittance and HBL's remittance services
- How to receive remittance without a bank account (Cash Over the Counter)
- Cash transaction limit: PKR 500,000 per transaction via CoC service
- Requirements: valid CNIC and Transaction Unique Reference No. / PIN code
- Branch locator reference for 1,600+ domestic branches
- Contact details: UAN 021-111-555-425 and email home.remittance@hbl.com

**Why Useful for RAG:**
Overseas Pakistanis frequently ask about how to send money home. This document contains very specific limits and procedures unique to HBL.

---

### 📄 HBL Document 3

| Field | Details |
|---|---|
| **Document Name** | HBL Islamic Current Account — Key Fact Sheet |
| **Source Link** | https://www.hbl.com/assets/documents/KFS_HBL_Islamic_Current_Account.pdf |
| **Document Type** | Key Fact Sheet (KFS) |
| **Topic** | Islamic banking account terms, security, and features |

**What It Contains:**
- Account security responsibilities of the customer
- ATM card, PIN, and cheque safety guidelines
- HBL's official policy: never asks for sensitive info (PIN, OTP, CVV) via call/SMS/email
- Safe custody instructions for internet banking credentials
- Procedure to close the Islamic Current Account
- Operations Manager contact procedure for account issues
- Shariah-compliance details for current account structure (Qard-based)

**Why Useful for RAG:**
Customers frequently ask about Islamic banking account rules and security policies. This KFS is highly specific to HBL's Islamic products.

---

### 📄 HBL Document 4

| Field | Details |
|---|---|
| **Document Name** | HBL @Work Conventional Accounts — Key Fact Sheet |
| **Source Link** | https://www.hbl.com/assets/documents/KFS_HBL@_Work_Conventional_Accounts.pdf |
| **Document Type** | Key Fact Sheet (KFS) |
| **Topic** | Salaried employee banking accounts — features and charges |

**What It Contains:**
- Account types available under HBL @Work program
- Salient features of each account tier
- Schedule of Bank Charges (SoBC) as applicable
- Fee structure: services, markup rates
- Designed for corporate/salaried employees enrolled through employer
- Terms valid as of 31 December 2021 (SoBC reference date)

**Why Useful for RAG:**
Corporate HR departments and employees frequently ask about employer-linked banking accounts. This document is very specific to HBL's corporate salary account program.

---

## 🏦 Bank 2: Meezan Bank

### About Meezan Bank
Meezan Bank is **Pakistan's first and largest dedicated Islamic bank**, founded in 2002. It operates on purely Shariah-compliant principles and offers a complete range of Islamic banking products including Mudarabah-based savings, home financing (Diminishing Musharakah), and Islamic auto loans. Headquartered in Karachi, it serves millions of customers across Pakistan.

---

### 📄 Meezan Document 1

| Field | Details |
|---|---|
| **Document Name** | Meezan Roshan Digital Account — FAQs |
| **Source Link** | https://www.meezanbank.com/wp-content/themes/mbl/downloads/overseas-pakistani-account/meezan-roshan-digital-account-FAQs.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Digital banking account for Non-Resident Pakistanis (NRPs) |

**What It Contains:**
- Account opening process for NRPs (online, no branch visit required)
- Eligible identity documents: CNIC, SNIC, NICOP, POC, Passport
- No initial deposit required to open account
- Account types: Qard-based Current and Mudarabah-based Savings
- Explanation of Mudarabah contract and profit-sharing model
- CDC (Central Depository Company) transfer facility via internet banking
- SWIFT code for wire transfers: MEZNPKKARDA
- Tax exemptions: withholding tax on cash withdrawals and fund transfers
- Pre-mature encashment process and schedule
- Multiple account opening allowed (one savings + one current)
- Record updation: via branch, email, or call center +92(21) 111-331-331/332

**Why Useful for RAG:**
This is one of the most information-rich documents in the dataset. Overseas Pakistanis have many specific questions about NRP accounts, tax rules, and Islamic finance terms.

---

### 📄 Meezan Document 2

| Field | Details |
|---|---|
| **Document Name** | Meezan Bank — Debit Card FAQs |
| **Source Link** | https://www.meezanbank.com/wp-content/themes/mbl/downloads/FAQ-debit-cards.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Debit card features, activation, and usage |

**What It Contains:**
- Debit card activation procedure
- Daily transaction and ATM withdrawal limits
- International usage policy
- Lost or stolen card reporting procedure
- Card replacement process
- POS transaction rules
- Security guidelines for cardholders

**Why Useful for RAG:**
Debit card queries are among the most common banking support questions. This document provides Meezan-specific policies that differ from other banks.

---

### 📄 Meezan Document 3

| Field | Details |
|---|---|
| **Document Name** | Meezan Digital Account Opening App — FAQs |
| **Source Link** | https://www.meezanbank.com/wp-content/themes/mbl/downloads/meezan-digital-account-FAQs.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Mobile app-based digital account opening |

**What It Contains:**
- Step-by-step digital account opening via mobile app
- Mobile number (SIM) verification process — OTP-based
- Biometric verification in-app and fallback branch visit (within 60 days)
- Documents required: CNIC photo/scan, source of income proof
- Transaction limits by account type:
  - Meezan Digital Remittance Account: PKR 1,000,000/month inward limit
  - Cash withdrawal: PKR 500,000/day max
  - Fund transfers: PKR 3,000,000 limit
- Zakat exemption declaration process
- Note: SIM verification NOT required for Meezan Digital Freelancer Account
- Cheque book and ATM card request process

**Why Useful for RAG:**
New-to-bank customers have many questions about digital onboarding. This FAQ answers very specific limit and document questions.

---

### 📄 Meezan Document 4

| Field | Details |
|---|---|
| **Document Name** | Meezan Bank — Freelancer Account FAQs |
| **Source Link** | https://www.meezanbank.com/wp-content/themes/mbl/downloads/freelancers-FAQs.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Dedicated banking account for freelancers |

**What It Contains:**
- Eligibility criteria for freelancer account
- NTN (National Tax Number) requirement and process
- Tax benefits for freelancers under Pakistani law
- Account features and transaction limits specific to freelancers
- Difference between freelancer account and regular account
- SIM verification waiver for freelancer account opening

**Why Useful for RAG:**
Pakistan has one of the fastest-growing freelancer economies globally. This is a highly relevant document for the target user base.

---

### 📄 Meezan Document 5

| Field | Details |
|---|---|
| **Document Name** | Meezan Roshan Apna Ghar — Home Financing FAQs |
| **Source Link** | https://www.meezanbank.com/wp-content/themes/mbl/downloads/Roshan-Apna-Ghar-FAQs.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Islamic home financing for overseas Pakistanis |

**What It Contains:**
- Eligibility for NRPs to apply for home financing from abroad
- Islamic financing structure (Diminishing Musharakah)
- Property types covered and cities available
- Documentation required from overseas applicants
- Down payment requirements
- Profit rates and repayment schedule
- Process to apply remotely without visiting Pakistan

**Why Useful for RAG:**
Home financing is a major need for overseas Pakistanis. This document covers a very specialized Islamic home loan product unavailable in general LLM training data.

---

## 🏦 Bank 3: Allied Bank Limited (ABL)

### About Allied Bank Limited
Allied Bank Limited (ABL) is one of Pakistan's **oldest and most established banks**, with a history dating back to 1942. It is the first Muslim bank established in the subcontinent. ABL has a vast branch network across Pakistan and offers comprehensive retail, corporate, and digital banking services. Its digital platform **myABL** is one of the leading mobile banking apps in Pakistan.

---

### 📄 ABL Document 1

| Field | Details |
|---|---|
| **Document Name** | myABL Digital Banking — FAQs |
| **Source Link** | https://www.abl.com/wp-content/uploads/2024/03/FAQs-myABL.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Mobile and internet banking registration, features, and usage |

**What It Contains:**
- Registration process for myABL internet and mobile banking
- Linking multiple ABL accounts to one myABL profile
- MPIN (Mobile PIN) creation and usage
- Fund transfer to registered and unregistered beneficiaries within Pakistan
- Bill payment: utility bills, school/university fees, internet, mobile top-up
- Biller and payee management (add from dashboard or after transaction)
- myABL Mastercard QR payment process and discount offers
- Biometric login: Touch ID / Face ID (iPhone only for Face ID)
- Account unblocking: call Allied Phone Banking 111-225-225
- Foreign national registration: requires in-person branch visit
- Non-financial services available even before full activation

**Why Useful for RAG:**
myABL is widely used across Pakistan. This FAQ covers very specific app features, limits, and procedures unique to Allied Bank.

---

### 📄 ABL Document 2

| Field | Details |
|---|---|
| **Document Name** | myABL Business Internet Banking — User Guidelines |
| **Source Link** | https://www.abl.com/wp-content/uploads/2023/03/myABL-Business-User-Guidelines.pdf |
| **Document Type** | User Guide / Guidelines Document |
| **Topic** | Business/corporate internet banking platform |

**What It Contains:**
- Business account internet banking setup and navigation
- IBFT (Inter Bank Fund Transfer) process for businesses
- Beneficiary creation and management for corporate users
- List of supported banks for transfers (MCB, HBL, Meezan, UBL, Faysal, etc.)
- Transaction authorization workflows for business accounts
- Security protocols for multi-user corporate accounts
- Payment processing for bulk transactions

**Why Useful for RAG:**
Business owners and corporate users frequently need guidance on internet banking platforms. This guide is very specific to ABL's business banking product.

---

## 🏦 Bank 4: Bank Alfalah

### About Bank Alfalah
Bank Alfalah is one of Pakistan's **leading private commercial banks**, established in 1997. It is backed by the Abu Dhabi Group and is known for its strong digital banking infrastructure. Bank Alfalah's **Alfalah RAPID** digital platform is one of Pakistan's most advanced digital banking ecosystems, offering full self-service banking online.

---

### 📄 Bank Alfalah Document 1

| Field | Details |
|---|---|
| **Document Name** | Alfalah RAPID — Self-Service Banking FAQs |
| **Source Link** | https://rapid.bankalfalah.com/alfalahrapidprod/assets/pdf/faqs/faqs_self_service_banking.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Online self-service banking portal features and usage |

**What It Contains:**
- What is Alfalah RAPID Banking and its services
- Prerequisites to use RAPID (registered account, email, cell number)
- Registration process: step-by-step via https://rapid.bankalfalah.com
- How to update email address and cell number (branch visit required)
- OTP (One Time Password) explanation: when and how it is generated
- Password reset and forgot password procedure
- WHT (Withholding Tax) statement: available for up to 3 years
- Security protocol: what to do if email is compromised
- SBP pilot approval for digital onboarding mentioned

**Why Useful for RAG:**
Alfalah RAPID is a widely used digital banking portal. Users frequently ask about registration and OTP issues — all covered here.

---

### 📄 Bank Alfalah Document 2

| Field | Details |
|---|---|
| **Document Name** | Alfalah Personal Loan — FAQs |
| **Source Link** | https://rapid.bankalfalah.com/alfalahrapidprod/assets/PDF/FAQs/FAQs_Personal_Loan.pdf |
| **Document Type** | FAQ Document |
| **Topic** | Personal loan product — eligibility, fees, and charges |

**What It Contains:**
- Application fee: No fee for applying
- Processing fee: PKR 4,000 OR 1.3% of loan amount (whichever applies)
- Age eligibility: Minimum 21 years, Maximum 60 years (salaried) / 65 years (SEB/SEP)
- Late payment charges: PKR 700 per month per installment
- Loan tenure and repayment structure
- Early repayment terms
- Required documents for personal loan application

**Why Useful for RAG:**
Personal loan queries are extremely common. This document contains very specific fee figures and age limits that a general LLM cannot accurately answer.

---

### 📄 Bank Alfalah Document 3

| Field | Details |
|---|---|
| **Document Name** | Bank Alfalah WhatsApp Banking — FAQs |
| **Source Link** | https://www.bankalfalah.com/wp-content/uploads/2021/08/FAQ-BAFL-WhatsApp-Banking-Channel.pdf |
| **Document Type** | FAQ Document |
| **Topic** | WhatsApp-based banking service |

**What It Contains:**
- What services are available via WhatsApp banking
- How to register for WhatsApp Banking
- Security features of the WhatsApp channel
- Types of transactions and inquiries supported
- How to initiate a session and navigate menu
- Limitations of WhatsApp banking vs full digital banking
- Helpline support integration

**Why Useful for RAG:**
WhatsApp banking is a growing trend in Pakistan. This document covers Bank Alfalah's specific WhatsApp channel — entirely unique information not in any LLM's training data.


---


## 📁 Dataset Summary Table

| # | Bank | Document Name | Topic | Link |
|---|---|---|---|---|
| 1 | HBL | Internet Payment Gateway FAQs | Merchant payment gateway | https://www.hbl.com/assets/documents/HBL_IPG_FAQs.pdf |
| 2 | HBL | Home Remittance FAQs | Overseas remittance | https://www.hbl.com/assets/documents/FAQS-_HBL_Fast_Transfer.pdf |
| 3 | HBL | Islamic Current Account KFS | Islamic account terms | https://www.hbl.com/assets/documents/KFS_HBL_Islamic_Current_Account.pdf |
| 4 | HBL | @Work Conventional Accounts KFS | Salaried employee accounts | https://www.hbl.com/assets/documents/KFS_HBL@_Work_Conventional_Accounts.pdf |
| 5 | Meezan | Roshan Digital Account FAQs | NRP digital account | https://www.meezanbank.com/wp-content/themes/mbl/downloads/overseas-pakistani-account/meezan-roshan-digital-account-FAQs.pdf |
| 6 | Meezan | Debit Card FAQs | Debit card usage | https://www.meezanbank.com/wp-content/themes/mbl/downloads/FAQ-debit-cards.pdf |
| 7 | Meezan | Digital Account Opening App FAQs | Mobile onboarding | https://www.meezanbank.com/wp-content/themes/mbl/downloads/meezan-digital-account-FAQs.pdf |
| 8 | Meezan | Roshan Apna Ghar FAQs | Islamic home financing | https://www.meezanbank.com/wp-content/themes/mbl/downloads/Roshan-Apna-Ghar-FAQs.pdf |
| 19 | Meezan | Solar Panel Financing FAQs | Green financing | https://www.meezanbank.com/wp-content/themes/mbl/downloads/solar-panel-FAQ/salaried-FAQs.pdf |
| 10 | Allied Bank | myABL Digital Banking FAQs | Mobile banking app | https://www.abl.com/wp-content/uploads/2024/03/FAQs-myABL.pdf |
| 11 | Allied Bank | myABL Business Banking Guide | Corporate internet banking | https://www.abl.com/wp-content/uploads/2023/03/myABL-Business-User-Guidelines.pdf |
| 12 | Bank Alfalah | RAPID Self-Service Banking FAQs | Digital banking portal | https://rapid.bankalfalah.com/alfalahrapidprod/assets/pdf/faqs/faqs_self_service_banking.pdf |
| 13 | Bank Alfalah | Personal Loan FAQs | Loan eligibility & charges | https://rapid.bankalfalah.com/alfalahrapidprod/assets/PDF/FAQs/FAQs_Personal_Loan.pdf |
| 14 | Bank Alfalah | WhatsApp Banking FAQs | WhatsApp channel | https://www.bankalfalah.com/wp-content/uploads/2021/08/FAQ-BAFL-WhatsApp-Banking-Channel.pdf |
| 15 | Bank Alfalah | Virtual Debit Card FAQs | Virtual card usage | https://www.bankalfalah.com/wp-content/uploads/2021/11/Bank-Alfalah-Virtual-Debit-Card-FAQs-Final.pdf |
| 16 | State Bank | Overall FAQs | Overall | https://www.sbp.org.pk/sbp_bsc/faqs.pdf |

---

## 📊 Dataset Statistics

| Property | Value |
|---|---|
| Total PDFs | 16 |
| Total Banks Covered | 5 |
| Primary Language | English |
| Domain | Pakistani Banking |
| Sub-domains | Digital Banking, Islamic Banking, Loans, Cards, Remittance, Freelancing |
| Source Type | Official Bank Websites (Primary Sources) |
| Data Type | FAQ + Policy + Key Fact Sheets + User Guides |
| Estimated Total Chunks (300–500 tokens) | ~150–250 chunks |
| Estimated Total Words | ~20,000–30,000 words |

---

## 💡 Sample RAG Questions This Dataset Can Answer

| Question | Relevant Document |
|---|---|
| What is the daily cash withdrawal limit for Meezan Digital Remittance Account? | Meezan Digital Account App FAQs |
| What SWIFT code should I use for Meezan Roshan account remittance? | Meezan Roshan Digital FAQs |
| What is the late payment charge on Alfalah Personal Loan? | Alfalah Personal Loan FAQs |
| Can a foreign national register for myABL digital banking? | ABL myABL FAQs |
| What documents are required for HBL Internet Payment Gateway? | HBL IPG FAQs |
| Is net metering included in Meezan solar panel financing? | Meezan Solar FAQs |
| How do I register for Bank Alfalah WhatsApp Banking? | Alfalah WhatsApp FAQs |
| What is the HBL remittance cash transaction limit? | HBL Remittance FAQs |
| Can I open multiple Meezan Roshan Digital Accounts? | Meezan Roshan Digital FAQs |
| What is the minimum age for Alfalah Personal Loan? | Alfalah Personal Loan FAQs |
| Can Federal Investment Bonds be redeemed before maturity? | State Bank FAQs |
| What is the withholding tax rate on PIBs? | State Bank FAQs |
| What did the Supreme Court rule about interest in Pakistan in 1999? | State Bank FAQs |
| Can Islamic banks charge penalty for late payment? | State Bank FAQs |
---


*Dataset curated for AI Knowledge Helper — RAG System Internship Project*  
*All documents sourced from official Pakistani bank websites — publicly available*
