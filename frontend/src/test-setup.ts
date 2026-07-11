// Vitest global setup: registers @testing-library/jest-dom matchers
// (toBeInTheDocument, toHaveTextContent, …) and cleans up the DOM after each test.
import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

afterEach(() => {
  cleanup();
});
