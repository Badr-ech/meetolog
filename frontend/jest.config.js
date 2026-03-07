/** @type {import('jest').Config} */
const config = {
  testEnvironment: "jsdom",
  roots: ["<rootDir>/tests"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
    "\\.module\\.css$": "identity-obj-proxy",
    "\\.css$": "identity-obj-proxy",
  },
  transform: {
    "^.+\\.tsx?$": [
      "ts-jest",
      {
        tsconfig: "tsconfig.json",
        jsx: "react-jsx",
      },
    ],
  },
  testPathIgnorePatterns: ["/node_modules/", "/.next/"],
  setupFilesAfterEnv: ["@testing-library/jest-dom"],
};

module.exports = config;
