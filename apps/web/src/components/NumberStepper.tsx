import { useState, useRef, useEffect, type ChangeEvent, type FocusEvent, forwardRef } from "react";

export interface NumberStepperProps {
  value: number | null;
  onChange: (value: number | null) => void;
  min?: number;
  max?: number;
  label?: string;
  disabled?: boolean;
  error?: string;
}

export const NumberStepper = forwardRef<HTMLDivElement, NumberStepperProps>(
  ({ value, onChange, min = 1, max = 100, label, disabled = false, error, ...props }, ref) => {
    const inputRef = useRef<HTMLInputElement>(null);
    const [inputValue, setInputValue] = useState<string>(value !== null ? String(value) : "");

    useEffect(() => {
      if (value !== null) {
        setInputValue(String(value));
      }
    }, [value]);

    const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      if (val === "" || /^\d+$/.test(val)) {
        setInputValue(val);
        if (val !== "") {
          const num = parseInt(val, 10);
          if (!isNaN(num) && num >= min && num <= max) {
            onChange(num);
          }
        } else {
          onChange(null as unknown as number);
        }
      }
    };

    const handleBlur = (e: FocusEvent<HTMLInputElement>) => {
      const val = e.target.value;
      if (val !== "") {
        const num = parseInt(val, 10);
        if (!isNaN(num)) {
          const clamped = Math.max(min, Math.min(max, num));
          if (clamped !== num) {
            setInputValue(String(clamped));
            onChange(clamped);
          }
        }
      }
    };

    const increment = () => {
      const current = value ?? min;
      if (current < max) {
        onChange(current + 1);
      }
    };

    const decrement = () => {
      const current = value ?? min;
      if (current > min) {
        onChange(current - 1);
      }
    };

    return (
      <div ref={ref} className="w-full max-w-xs" {...props}>
        {label && <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>}
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={decrement}
            disabled={disabled || (value !== null && value <= min)}
            className="w-12 h-12 rounded-lg bg-gray-100 text-gray-900 text-2xl font-medium hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Kurangi"
          >
            −
          </button>
          <input
            ref={inputRef}
            type="number"
            value={inputValue}
            onChange={handleChange}
            onBlur={handleBlur}
            min={min}
            max={max}
            disabled={disabled}
            className={`flex-1 h-12 text-center text-xl font-medium border-2 rounded-lg
              ${error ? "border-red-500" : "border-gray-300"}
              focus:outline-none focus:ring-2 focus:ring-blue-500
              disabled:bg-gray-50 disabled:text-gray-500`}
            aria-label={label}
            aria-invalid={error ? "true" : "false"}
            inputMode="numeric"
          />
          <button
            type="button"
            onClick={increment}
            disabled={disabled || (value !== null && value >= max)}
            className="w-12 h-12 rounded-lg bg-gray-100 text-gray-900 text-2xl font-medium hover:bg-gray-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-blue-500"
            aria-label="Tambah"
          >
            +
          </button>
        </div>
      </div>
    );
  }
);

NumberStepper.displayName = "NumberStepper";