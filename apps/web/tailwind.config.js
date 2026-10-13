import {
  colors,
  palette,
  fontFamily,
  fontSize,
  fontWeight,
  radius,
} from "./src/design/tokens.ts";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Brand + semantic tokens (see src/design/tokens.ts).
        brand: colors.surface.brand,
        "brand-soft": colors.surface.brandSoft,
        "brand-deep": palette.brandDeep,
        "progress-track": palette.progressTrack,
        "progress-fill": palette.progressFill,
        success: colors.status.success,
        "success-soft": palette.successSoft,
        warning: colors.status.warning,
        "warning-soft": palette.warningSoft,
        danger: colors.status.danger,
        "danger-soft": palette.dangerSoft,
        "text-primary": colors.text.primary,
        "text-inverse": colors.text.inverse,
        "text-muted": colors.text.muted,
        "surface-subtle": colors.surface.subtle,
        "border-default": colors.border.default,
        "border-strong": colors.border.strong,
      },
      fontFamily: {
        sans: fontFamily.sans,
        display: fontFamily.display,
        alt: fontFamily.alt,
      },
      fontSize: {
        xs: `${fontSize.xs}px`,
        sm: `${fontSize.sm}px`,
        base: `${fontSize.base}px`,
        lg: `${fontSize.lg}px`,
        xl: `${fontSize.xl}px`,
        "2xl": `${fontSize["2xl"]}px`,
        "3xl": `${fontSize["3xl"]}px`,
        display: `${fontSize.display}px`,
      },
      fontWeight: {
        normal: `${fontWeight.normal}`,
        medium: `${fontWeight.medium}`,
        semibold: `${fontWeight.semibold}`,
      },
      borderRadius: {
        sm: `${radius.sm}px`,
        md: `${radius.md}px`,
        lg: `${radius.lg}px`,
        pill: `${radius.pill}px`,
      },
      spacing: {
        // Horizontal page gutter. Fluid so 402px-wide screens keep the Figma
        // 30px exactly, while narrower phones scale it down (clamp) instead of
        // squeezing the content column and re-wrapping text.
        gutter: "clamp(18px, 7.46vw, 30px)",
      },
      maxWidth: {
        device: "402px",
      },
    },
  },
  plugins: [],
};
