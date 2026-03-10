// `CI` is a special variable which is automatically set to true in the CI by Github and is always
// false by default in local. This value is never manually set or changed
export const CI = process.env.CI;

export const BASE_URL = process.env.BASE_URL || "http://localhost:5005";
console.info("Running inspector e2e tests on", BASE_URL);
