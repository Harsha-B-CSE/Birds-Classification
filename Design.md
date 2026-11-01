# 🎨 UI/UX Design System Specification
## Project: Birds Classification & Avian Intelligence System (Feathered Insights)
### Theme: Natural Metallic (Biophilic Luxury)

---

## 1. Design Philosophy: "Natural Metallic"

The visual aesthetic of **Feathered Insights** is built upon the concept of **Natural Metallic**—a harmonious fusion of organic forest depth, iridescent avian plumage, and warm burnished metals (aged copper, weathered bronze, and brushed gold).

### Core Aesthetic Tenets
1. **Atmospheric Depth**: Deep, organic forest darkness (`hsl(150, 20%, 8%)`) replaces stark black or sterile white backgrounds. Layered glassmorphism creates natural visual hierarchy without visual clutter.
2. **Plumage-Inspired Accents**: Vibrant biological highlights—such as **Feather Teal** and **Burnished Copper**—draw immediate focus to key classification metrics and confidence meters.
3. **Tactile & Responsive Feedback**: High-performance micro-animations (scan beams, soft metallic glows, smooth probability bar infills) evoke scientific precision and modern elegance.
4. **Zero Generic AI Slop**: Custom-crafted HSL color palettes, tailored typography from Google Fonts (*Outfit* & *Inter*), and custom-curated status badges eliminate generic framework styling.

---

## 2. Color Palette & CSS Design Tokens

```css
:root {
  /* ================= Surface & Backgrounds ================= */
  --bg-deep-forest:       hsl(150, 20%, 8%);      /* #0b140e - Base page background */
  --bg-card-glass:        rgba(22, 38, 28, 0.72); /* Translucent moss glass */
  --bg-card-glass-hover:  rgba(28, 48, 36, 0.85); /* Hover card elevation */
  --bg-surface-elevated:  hsl(152, 22%, 12%);     /* #17271e - Modals and inner containers */
  --bg-input-field:       rgba(12, 22, 16, 0.8);  /* Recessed input background */

  /* ================= Metallic Accent Palette ================= */
  --accent-copper:        hsl(28, 65%, 45%);      /* #bd6328 - Primary button & hero emphasis */
  --accent-copper-glow:   rgba(189, 99, 40, 0.35);/* Soft metallic halo glow */
  --accent-gold:          hsl(43, 70%, 52%);      /* #e2b027 - Secondary metallic accent & rank 1 */
  --accent-gold-soft:     rgba(226, 176, 39, 0.2);/* Pill background */
  --accent-teal:          hsl(175, 50%, 45%);     /* #39ad9e - High-confidence badge & genetics */
  --accent-teal-glow:     rgba(57, 173, 158, 0.3);/* Scan line effect */

  /* ================= Borders & Separators ================= */
  --border-copper-subtle: rgba(189, 99, 40, 0.28);/* Subtle card border */
  --border-glass:         rgba(255, 255, 255, 0.08);/* Delicate glass highlight */
  --border-active:        hsl(28, 75%, 55%);      /* Active input / focused element */

  /* ================= Typography & Text ================= */
  --text-ivory-primary:   hsl(40, 30%, 92%);      /* #f4f0e8 - Primary headlines & body */
  --text-sage-secondary:  hsl(140, 12%, 68%);     /* #a2ada6 - Subtitles, metadata, taxonomy */
  --text-muted:           hsl(140, 8%, 48%);      /* #707b73 - Timestamps & captions */

  /* ================= Status Indicators ================= */
  --status-success:       hsl(142, 60%, 45%);     /* Bird verified & high confidence */
  --status-warning:       hsl(38, 92%, 50%);      /* Low confidence / borderline energy */
  --status-error:         hsl(0, 68%, 52%);       /* Non-bird rejected / OOD */
  --status-info:          hsl(198, 80%, 48%);     /* Informational tags */

  /* ================= Shadows & Radii ================= */
  --radius-sm: 8px;
  --radius-md: 14px;
  --radius-lg: 22px;
  --radius-full: 9999px;

  --shadow-glass:         0 8px 32px 0 rgba(0, 0, 0, 0.42);
  --shadow-metallic-glow: 0 0 24px -4px var(--accent-copper-glow);
  --backdrop-blur:        blur(16px);
}
```

---

## 3. Typography System

The typography pairs a characterful geometric display font for avian taxonomy with a high-legibility sans-serif for dense biological data:

- **Primary Display / Headlines**: `Outfit`, sans-serif (Google Fonts)
  - Weights: `600 (SemiBold)`, `700 (Bold)`
  - Letter-spacing: `-0.02em` on titles
- **Body & Scientific Text**: `Inter`, sans-serif (Google Fonts)
  - Weights: `400 (Regular)`, `500 (Medium)`
  - Line-height: `1.6` for optimal readability in long encyclopedic passages
- **Genomic & Taxonomic Tags**: `JetBrains Mono`, monospace (Google Fonts)
  - Weights: `500 (Medium)`
  - Used for NCBI Taxon IDs, Chromosome counts ($2n=76$), and tensor processing latency indicators.

---

## 4. Component Blueprint & UI Layouts

### 4.1 Master Layout Structure
The application adopts an unauthenticated, single-purpose workflow organized around three distinct pages:

1. **`/predict/` (Primary Identification Hub)**:
   - Dynamic Drag-and-Drop Image Dropzone with real-time preview.
   - 3-Check Pipeline Status Pills (Live progression: Detector → Classifier → OOD Gate).
   - Species Hero Classification Card with IUCN Conservation Pill.
   - Top-5 Confidence Distribution Bar Chart.
   - Biological Deep-Dive Accordion (Taxonomy Tree, Lifespan Chart.js, Genetics & Habitat).
2. **`/history/` (Personal Avian Field Log)**:
   - Grid of past identifications stored locally and synced anonymously via Firestore.
   - Filtering by date, confidence rating, and conservation status.
3. **`/about/` (Methodology & Model Architecture)**:
   - Swin Transformer benchmark metrics (Top-1, Top-5, Confusion Matrix).
   - OOD Energy threshold explanation and CUB-200-2011 dataset attribution.

---

### 4.2 Interactive Components

#### A. Avian Dropzone Uploader
- **Resting State**: Translucent moss card with subtle oxidized copper dashed border. Centered feather icon, with call-to-action: *"Drop bird photograph here or click to browse"*.
- **Drag-Over State**: Border turns solid burnished bronze (`--accent-copper`) with a pulsing ambient glow (`--shadow-metallic-glow`).
- **Processing State**: 
  - Image preview appears with a high-tech **Feather Teal scan beam** traveling vertically across the image.
  - Sequential checkmark indicators update in real time:
    - `[✓] Bird Presence Confirmed (Check 1: 99.4%)`
    - `[✓] Feature Extraction Complete (Check 2: Swin Transformer)`
    - `[✓] Species In-Distribution Verified (Check 3: Energy Score -12.4)`

#### B. Species Result Hero Card
```
+-----------------------------------------------------------------------------------------+
|  [THUMBNAIL]   COMMON NAME: Indian Peafowl                   [IUCN: LEAST CONCERN 🟢]  |
|                SCIENTIFIC:  Pavo cristatus                                              |
|                CONFIDENCE:  94.3%  ==================================[ 94.3% ]         |
|                FAMILY:      Phasianidae  |  ORDER: Galliformes                          |
+-----------------------------------------------------------------------------------------+
|  TOP-5 PREDICTION DISTRIBUTION:                                                        |
|  1. Indian Peafowl       [94.3%] ====================================================   |
|  2. Green Peafowl        [ 3.1%] ==                                                     |
|  3. Congo Peafowl        [ 1.2%] =                                                      |
|  4. Great Argus          [ 0.8%] =                                                      |
|  5. Ring-necked Pheasant [ 0.6%] =                                                      |
+-----------------------------------------------------------------------------------------+
```

#### C. Biological Deep-Dive Tabs (Chart.js Integration)
- **Lifespan Comparison Chart**: Horizontal bar chart rendered via Chart.js comparing Wild Lifespan (10–25 yrs) vs. Captive Lifespan (up to 40 yrs) in dual-toned copper/gold bars against moss background.
- **Taxonomic Hierarchy Tree**: Clean nested pill badges connecting Kingdom (*Animalia*) → Phylum (*Chordata*) → Class (*Aves*) → Order (*Galliformes*) → Family (*Phasianidae*) → Genus (*Pavo*) → Species (*P. cristatus*).
- **Genomic & Molecular Box**: Monospace badges displaying NCBI Taxon ID (with external GenBank link), Chromosome diploid number ($2n=76$), and approximate genome size.

---

## 5. Error & Out-of-Distribution (OOD) States

Rejections must feel informative and scientific, rather than punitive:

### 1. Non-Bird Input (Check 1 Failure)
- **Visual**: Warm amber border, subtle bird silhouette with gentle slash icon.
- **Copy**: *"No Avian Subject Detected. Our primary detector did not observe characteristic bird morphology (feathers, bill, or wings) in this photograph. Please provide a clear, unobstructed shot of a bird."*

### 2. Unknown / Novel Species (Check 2 & 3 OOD Trigger)
- **Visual**: Feather teal card with pulsing compass icon.
- **Copy**: *"Novel or Uncataloged Avian Species. The model confirms an avian subject is present; however, its visual features do not match any of the 200+ species currently in our high-confidence catalog. This helps prevent inaccurate identifications."*

---

## 6. Micro-Interactions & Motion Tokens

| Animation | Duration | Easing | Purpose |
|---|---|---|---|
| **Feather Scan Beam** | $1.8\text{ s}$ | `cubic-bezier(0.4, 0, 0.2, 1)` | Vertical scanning shimmer during inference |
| **Probability Bar Fill**| $0.9\text{ s}$ | `cubic-bezier(0.16, 1, 0.3, 1)` | Smooth fill animation on result display |
| **Glass Card Hover** | $0.25\text{ s}$| `ease-out` | Transform `translateY(-3px)`, glow bloom |
| **Accordion Expand** | $0.35\text{ s}$| `cubic-bezier(0.25, 1, 0.5, 1)` | Smooth slide-down of biological dossier |

---

## 7. Responsive Breakpoint Rules

- **Mobile Viewport (`360px – 767px`)**:
  - Single-column flow: Upload zone occupies full width; result card stacks thumbnail vertically above species metadata.
  - Chart.js renders with compact aspect ratio (2:1).
  - Floating bottom bar for rapid "Identify Another Bird" action.
- **Tablet Viewport (`768px – 1023px`)**:
  - Two-column split layout: Left side sticky upload & preview; right side scrollable biological dossier.
- **Desktop Viewport (`1024px+`)**:
  - Max container width capped at `1240px` with generous horizontal margins for an expansive, desktop-grade dashboard feel.
