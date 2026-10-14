/**
 * Design tokens extracted from the Figma file
 * "Joints UGM 2026 UI Section" (key 4ZBJiFe0AaKHDmM0C6qxdR).
 *
 * Every value below is read directly from the Figma node JSON (fills, strokes,
 * text styles, cornerRadius, itemSpacing) — not estimated from screenshots.
 * The source of truth stays the Figma file; when the design changes, re-extract
 * and update this file. `tailwind.config.js` consumes these tokens.
 */

/** Raw palette as sampled from Figma solid fills. */
export const palette = {
  black: "#000000",
  white: "#FFFFFF",
  /** Primary brand colour (teal). */
  brand: "#0F766E",
  /** Muted brand fill used for soft backgrounds. */
  brandSoft: "#B0E4E0",
  /** Success / low-concern indicators. */
  success: "#16A34A",
  /** Warning / amber indicators. */
  warning: "#F59E0B",
  /** Danger / high-concern indicators. */
  danger: "#C83E5C",
  /** Soft warning surface. */
  warningSoft: "#FEF3C7",
  /** Soft danger surface. */
  dangerSoft: "#FFD7E0",
  /** Soft success surface, for low-concern results. */
  successSoft: "#DCFCE7",
  /** Soft accent surface (yellow). */
  accentSoft: "#FFE683",
  /** Brand gradient end stop (Figma linear gradient #0F766E → #115E59). */
  brandDeep: "#115E59",
  /** Progress-bar track + fill sampled from the question screens. */
  progressTrack: "#E1E1E1",
  progressFill: "#B3A8A8",
  neutral50: "#FFFFFF",
  neutral100: "#E3E3E3",
  neutral200: "#E1E1E1",
  neutral300: "#C4C4C4",
  neutral400: "#B4A9A9",
  neutral500: "#CCD2E3",
} as const;

/**
 * theme change is a single edit.
 */
export const colors = {
  text: {
    primary: palette.black,
    inverse: palette.white,
    muted: palette.neutral400,
  },
  surface: {
    default: palette.white,
    subtle: palette.neutral100,
    brand: palette.brand,
    brandSoft: palette.brandSoft,
  },
  border: {
    default: palette.neutral500,
    strong: palette.neutral300,
    brand: palette.brand,
  },
  status: {
    success: palette.success,
    warning: palette.warning,
    danger: palette.danger,
  },
} as const;

/**
 * Font families. `Geist` is the UI/body face, `Sora` the display/logo face and
 * `Plus Jakarta Sans` a secondary body face. The `--font-*` fallbacks keep
 * rendering sane before the webfonts load.
 */
export const fontFamily = {
  sans: ['"Geist Variable"', "Geist", "system-ui", "sans-serif"],
  display: ['"Sora Variable"', "Sora", "system-ui", "sans-serif"],
  alt: ['"Plus Jakarta Sans Variable"', '"Plus Jakarta Sans"', "system-ui", "sans-serif"],
} as const;

/** Font sizes (px) observed in Figma, mapped to a type scale. */
export const fontSize = {
  xs: 12,
  sm: 14,
  base: 16,
  lg: 18,
  xl: 20,
  "2xl": 22,
  "3xl": 24,
  display: 40,
} as const;

/** Font weights observed: 400 body, 500 medium, 600 semibold. */
export const fontWeight = {
  normal: 400,
  medium: 500,
  semibold: 600,
} as const;

/** Corner radii (px) observed, mapped to semantic names. */
export const radius = {
  sm: 4,
  md: 12,
  /** Primary button / card radius (Figma uses 16 for buttons). */
  lg: 16,
  /** Large pill radius. */
  pill: 100,
} as const;

/** Spacing scale (px) derived from Figma itemSpacing and frame gutters. */
export const spacing = {
  xs: 4,
  sm: 8,
  md: 10,
  /** Default horizontal page gutter measured across frames. */
  gutter: 30,
  lg: 16,
  xl: 24,
  "2xl": 32,
} as const;

/** The reference device frame the designs are drawn on (iPhone 16/17 Pro). */
export const deviceFrame = {
  width: 402,
  height: 874,
} as const;
