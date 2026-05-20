import { fixupPluginRules } from "@eslint/compat";
import pluginJs from "@eslint/js";
import globals from "globals";
import tseslint from "typescript-eslint";
import prettier from "eslint-config-prettier";
import promise from "eslint-plugin-promise";
import importPlugin from "eslint-plugin-import";
import vitest from "@vitest/eslint-plugin";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import playwright from "eslint-plugin-playwright";

// These are the rules that should be applied to every file (or group of files)
const sharedRules = {
  "@typescript-eslint/no-unused-vars": [
    "error",
    {
      args: "all",
      argsIgnorePattern: "^_",
      caughtErrors: "all",
      caughtErrorsIgnorePattern: "^_",
      destructuredArrayIgnorePattern: "^_",
      varsIgnorePattern: "^_",
      ignoreRestSiblings: true,
    },
  ],
  "@typescript-eslint/no-empty-function": "error",
  "@typescript-eslint/consistent-type-imports": "error",
  "@typescript-eslint/no-floating-promises": "error",
  "@typescript-eslint/no-explicit-any": "error",
  "@typescript-eslint/no-unsafe-call": "error",
  "@typescript-eslint/no-unsafe-member-access": "error",
  "@typescript-eslint/no-unsafe-return": "error",
  "@typescript-eslint/no-unsafe-assignment": "error",
  "@typescript-eslint/no-unsafe-argument": "error",
  "@typescript-eslint/no-unsafe-enum-comparison": "off",
  "@typescript-eslint/require-await": "error",
  "no-constant-condition": "error",
  "no-case-declarations": "warn",
  "no-constant-binary-expression": "off",
  "no-extra-boolean-cast": "off",
  "@typescript-eslint/no-unused-expressions": "off",
};

export default [
  {
    // Global ignores
    ignores: [
      "./node_modules/**",
      "./dist/**",
      "./build/**",
      "./coverage/**",
      "./e2e/playwright-report/",
      "./e2e/test-results/",
      "./e2e/e2e/test-results/**",
      "./*.config.js",
      "./*.config.mjs",
      "./*.config.ts",
      "./.eslintrc.js",
      "./vite.config.ts",
      "./tailwind.config.ts",
      "./postcss.config.js",
      "./components.json",
      "./sonar-project.properties",
      "./.husky/**",
      "./.github/**",
      "./.cursor/**",
      "./.sonarlint/**",
      "./.vscode/**",
      "./public/**",
      "./scripts/**",
    ],
  },
  pluginJs.configs.recommended,
  ...tseslint.configs.recommendedTypeChecked,
  promise.configs["flat/recommended"],
  prettier,
  {
    name: "base-typescript",
    files: ["**/*.ts", "**/*.tsx"],
    languageOptions: {
      parserOptions: {
        tsconfigRootDir: import.meta.dirname,
        ecmaVersion: "latest",
        sourceType: "module",
        project: [
          "./tsconfig.json",
          "./e2e/tsconfig.json",
        ],
      },
      globals: {
        ...globals.node,
        ...globals.es2021,
      },
    },
    plugins: { import: importPlugin },
    rules: {
      ...sharedRules,
      "import/no-duplicates": ["error", { "prefer-inline": true }],
    },
  },
  {
    name: "vitest-tests",
    files: ["./*.test.tsx", "./*.test.ts", "./*.spec.tsx", "./*.spec.ts"],
    plugins: { vitest },
    rules: {
      ...sharedRules,
      ...vitest.configs.recommended.rules,
    },
  },
  {
    name: "frontend-react",
    files: [
      "**/*.ts",
      "**/*.tsx",
    ],
    ignores: [
      "e2e/**/*.ts",
      "e2e/**/*.tsx",
    ],
    plugins: {
      react,
      "react-hooks": fixupPluginRules(reactHooks),
      "react-refresh": reactRefresh,
    },
    rules: {
      ...sharedRules,
      ...react.configs.flat.recommended.rules,
      ...reactHooks.configs.recommended.rules,
      "react-refresh/only-export-components": [
        "warn",
        { allowConstantExport: true },
      ],
      "no-prototype-builtins": "off",
      "@typescript-eslint/no-unsafe-assignment": "error",
      "@typescript-eslint/no-unsafe-member-access": "error",
      "@typescript-eslint/no-unsafe-call": "error",
      "@typescript-eslint/no-unsafe-return": "error",
      "@typescript-eslint/no-unsafe-argument": "error",
      "@typescript-eslint/unbound-method": "error",
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/no-explicit-any": "error",
      "react/react-in-jsx-scope": "off",
      "react/no-unescaped-entities": "off",
      "react/prop-types": "off",
      "react/display-name": "off",
      "no-restricted-syntax": [
        "error",
        {
          selector:
            "CallExpression[callee.name='useStore'][arguments.length=1] > Identifier.arguments:first-child, CallExpression[callee.name='useStore'][arguments.length=1] > MemberExpression.arguments:first-child",
          message:
            "useStore from @tanstack/react-store requires a selector function as the second argument to prevent unnecessary re-renders. Use: useStore(store, (state) => state.value)",
        },
      ],
    },
  },
  {
    name: "e2e-specific",
    files: ["e2e/**/*.ts", "e2e/**/*.tsx"],
    plugins: { playwright },
    rules: {
      ...sharedRules,
      ...playwright.configs["flat/recommended"].rules,
      "playwright/expect-expect": "off",
      "playwright/no-conditional-in-test": "off",
      "no-restricted-imports": [
        "error",
        {
          patterns: [
            {
              group: ["../actions/ui/*"],
              message: "Import UI actions only from @e2e/ui-actions.",
            },
            {
              group: ["../flows/*", "!../flows/index"],
              message: "Import flows only from the index.ts barrel file.",
            },
            {
              group: ["@e2e/ui-actions/*"],
              message: "Import UI actions only from @e2e/ui-actions.",
            },
            {
              group: ["@e2e/flows/*"],
              message: "Import flows only from @e2e/flows.",
            },
          ],
        },
      ],
    },
  },
];
