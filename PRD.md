# 📋 Product Requirements Document (PRD)
## Project: Birds Classification & Avian Intelligence System (Feathered Insights)

---

## 1. Executive Summary

**Birds Classification (Feathered Insights)** is an AI-powered avian identification and biodiversity intelligence platform. The application allows users—ranging from amateur birders and student naturalists to professional field ornithologists—to upload any bird image and receive an instant, accurate species classification accompanied by an encyclopedic biological profile in under **2 seconds**.

Unlike standard computer vision models that hallucinate on non-avian objects or unfamiliar wildlife, Feathered Insights employs a **three-tier Out-of-Distribution (OOD) verification gate** to guarantee prediction integrity. The system pairs state-of-the-art vision transformers (**Swin Transformer**) with live scientific enrichment APIs (Wikipedia, iNaturalist, NCBI Taxonomy, GBIF), delivering a zero-friction, public-facing web experience hosted on Firebase and Cloud Run.

---

## 2. Problem Statement & Market Context

1. **Context Drift & Hallucination in Field AI**: Most bird classification tools classify *any* input into one of their predefined training classes, erroneously identifying cars, pet dogs, or blurred vegetation as birds with high false confidence.
2. **Lack of Fine-Grained Domain Context**: Identifying a species is only the first step. Observers need immediate access to taxonomy, conservation status (IUCN), lifespan, native range, and genetics without having to cross-reference multiple scholarly databases.
3. **High Barrier to Entry**: Existing research platforms often mandate user registration, subscriptions, or complex parameter tuning, discouraging rapid mobile field usage.
4. **Latency & Regional Gaps**: Mobile field identification demands sub-2-second inference, while mainstream datasets (such as CUB-200-2011) underrepresent regional birds (such as endemic Indian subcontinent species).

---

## 3. Goals & Success Metrics

### 3.1 Primary Goals
- Deliver instant species identification across **200+ global & regional bird species** (CUB-200-2011 benchmark + 25 Indian species expansion).
- Implement a **3-check validation gate** to filter non-birds, low-confidence captures, and unseen out-of-distribution species.
- Provide a rich, automated scientific profile (taxonomy tree, lifespan chart, IUCN status, NCBI genomic taxon).
- Ensure a completely open, **frictionless access model** (no mandatory account creation or login wall).
- Guarantee an end-to-end response latency of **<2.0 seconds** (target ~540ms on cache hit).

### 3.2 Success Metrics & KPIs
| Metric | Target | Measurement Method |
|---|---|---|
| **Top-1 Accuracy** | $\ge 92.0\%$ | Validation set evaluation on CUB-200-2011 |
| **Top-5 Accuracy** | $\ge 98.0\%$ | Validation set evaluation |
| **OOD Rejection Rate** | $\ge 95.0\%$ | Rejection of non-bird & unseen species test sets |
| **End-to-End Latency** | $< 2.0\text{ s}$ | P95 server response time under regular network |
| **Cached Species Fetch** | $< 100\text{ ms}$ | Firestore `bird_info_cache` read speed |
| **Mobile Responsiveness** | $100\%$ | Zero horizontal overflow on 360px–1440px screens |

---

## 4. User Personas

### Persona A: Dr. Ananya Sharma (Field Ornithologist & Researcher)
- **Profile**: 34, conducts avian biodiversity surveys across the Western Ghats.
- **Pain Point**: Needs rapid validation of subtle morphological differences between closely related species; requires scientific names, IUCN status, and NCBI taxon links.
- **Need**: High confidence thresholds, top-5 probability distribution, and explicit OOD rejection when capturing an uncataloged specimen.

### Persona B: Rahul Verma (Casual Birdwatcher & Nature Enthusiast)
- **Profile**: 22, college student who photographs birds in urban parks and sanctuaries using a smartphone.
- **Pain Point**: Frustrated by apps that force logins or fail to identify native Indian birds.
- **Need**: Instant drag-and-drop upload, aesthetic visual feedback, clear habitat and lifespan statistics.

---

## 5. User Journeys & Core Workflows

### Journey 1: Successful Field Identification
1. **Entry**: User visits `https://birds-classification.web.app` (or local instance).
2. **Input**: User drags an image or taps "Upload Bird Photo" from their smartphone/browser.
3. **Processing Feedback**: Interface displays an animated radar/scan beam with real-time pipeline status (*"Checking bird presence → Analyzing plumage features → Fetching scientific dossier"*).
4. **Result Delivery**:
   - Primary species name (Common & Scientific) + confidence badge.
   - Top-5 prediction distribution bar chart.
   - IUCN conservation status indicator (e.g., *Least Concern*, *Vulnerable*).
   - Biological Dossier: Habitat, wild vs. captive lifespan (Chart.js), DNA chromosome count, and NCBI taxonomy link.
5. **Session History**: Result automatically syncs to device-keyed local/Firestore history without requiring an account.

### Journey 2: Out-of-Distribution / Non-Bird Input
1. **Input**: User uploads a photo of a domestic cat or an automobile.
2. **Check 1 Trigger**: Binary Bird Detector (MobileNetV3) calculates bird probability $< 60\%$.
3. **Response**: Pipeline immediately halts further inference.
4. **Feedback**: Friendly, informative card: *"🚫 No Bird Detected. Our sensor did not recognize avian features in this photo. Please upload a clear photo of a bird."*

### Journey 3: Unseen / Novel Species
1. **Input**: User uploads an image of an exotic bird outside the 200+ trained classes.
2. **Check 2 Trigger**: Energy-based OOD metric exceeds the calibrated rejection threshold ($E > \tau$).
3. **Response**: System abstains from guessing.
4. **Feedback**: *"🔍 Unknown Avian Species. The model confirms this is a bird, but its features do not match any cataloged species in our current database with reliable confidence."*

---

## 6. Functional Requirements

### 6.1 Image Ingestion & Preprocessing
- **FR-01**: Support JPEG, PNG, WEBP, and HEIC (converted) formats up to 15MB.
- **FR-02**: Client-side image downscaling/compression to $< 1.0\text{ MB}$ and $224 \times 224$ (or $384 \times 384$) bicubic tensor normalization prior to upload.
- **FR-03**: Automatic EXIF orientation normalization.

### 6.2 Inference & Gating Engine
- **FR-04**: **Check 1 (Bird Detector)**: Fast binary classification (MobileNetV3 ONNX) to verify bird presence ($\ge 60\%$).
- **FR-05**: **Check 2 (Species Classification)**: Swin Transformer forward pass yielding logits across all classes.
- **FR-06**: **Check 3 (Energy OOD Gate)**: Compute Helmholtz free energy $E(x) = -T \cdot \log \sum \exp(z_i / T)$. Reject if energy score exceeds calibrated threshold.
- **FR-07**: **Confidence Floor**: Top-1 probability must exceed $40.0\%$; otherwise, return an "insufficient evidence" advisory.

### 6.3 Scientific Enrichment Engine
- **FR-08**: Auto-fetch Wikipedia summary, taxonomy hierarchy (Aves → Order → Family → Genus → Species), IUCN Red List status, habitat, lifespan data, and NCBI taxonomy ID.
- **FR-09**: Caching: Query Firestore `bird_info_cache` first. On cache miss, fetch via external REST APIs asynchronously and write back to Firestore.

### 6.4 History & Anonymous Personalization
- **FR-10**: Generate a persistent UUID in browser `localStorage` (Device ID).
- **FR-11**: Store prediction history under Firestore collection `predictions/{device_id}/history` (species, confidence, thumbnail, timestamp).
- **FR-12**: Allow users to browse their past predictions on `/history/` and clear history on demand.

---

## 7. Non-Functional Requirements

- **NFR-01 (Latency)**: End-to-end response time $\le 2000\text{ ms}$ on standard 4G connections. Target $\le 600\text{ ms}$ for cached species.
- **NFR-02 (Storage Bounds)**: Local training/dev machine cache (`D:\BirdsCache\`) must strictly enforce a **5.0 GB ceiling** with automated LRU eviction.
- **NFR-03 (High Availability)**: Serverless deployment via Google Cloud Run with auto-scaling down to 0 instances and up during demand spikes.
- **NFR-04 (Security)**: Cloud service account keys (`serviceAccountKey.json`) must never be exposed to clients or checked into source control.
- **NFR-05 (Accessibility & Aesthetics)**: Comply with WCAG 2.1 AA standards; adhere to the "Natural Metallic" theme token specifications.

---

## 8. Out of Scope (Version 1.0)

- Bioacoustic / bird chirp sound classification (reserved for v2.0).
- Multi-bird object detection and bounding box localization in a single frame.
- User accounts, social followings, or community comment threads.
- Direct hardware edge deployment on microcontrollers (Raspberry Pi/Coral).
