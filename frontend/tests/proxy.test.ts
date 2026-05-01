/**
 * Tests for the Next.js rewrite-based proxy routing defined in next.config.js.
 *
 * next.config.js uses `rewrites()` to forward /api/:path* requests to the
 * FastAPI backend.  These tests validate the configuration object directly
 * since there is no standalone proxy.ts file in this project.
 *
 * NOTE: Next.js 16 does not yet ship public, stable testing utilities such as
 * `unstable_getResponseFromNextConfig` or `unstable_doesProxyMatch`.  As a
 * pragmatic alternative we import the config and assert on the rewrite rules.
 * If those utilities become available in a future Next.js release, these tests
 * should be migrated accordingly.
 */

// eslint-disable-next-line @typescript-eslint/no-var-requires
const nextConfig = require("../next.config.js");

interface RewriteRule {
  source: string;
  destination: string;
}

describe("next.config.js proxy rewrites", () => {
  let rewrites: RewriteRule[];

  beforeAll(async () => {
    const result = nextConfig.rewrites();
    rewrites = result instanceof Promise ? await result : result;
  });

  it("returns at least one rewrite rule", () => {
    expect(Array.isArray(rewrites)).toBe(true);
    expect(rewrites.length).toBeGreaterThanOrEqual(1);
  });

  it("proxies /api/:path* to the backend", () => {
    const apiRule = rewrites.find((r) => r.source === "/api/:path*");
    expect(apiRule).toBeDefined();
    expect(apiRule!.destination).toContain(":path*");
  });

  it("defaults backend URL to http://localhost:8000 when env is unset", () => {
    const saved = process.env.API_URL;
    const savedPublic = process.env.NEXT_PUBLIC_API_URL;
    delete process.env.API_URL;
    delete process.env.NEXT_PUBLIC_API_URL;

    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const freshConfig = require("../next.config.js");
    const rules: RewriteRule[] | Promise<RewriteRule[]> = freshConfig.rewrites();
    const resolved = rules instanceof Promise ? rules : Promise.resolve(rules);

    return resolved.then((r) => {
      const apiRule = r.find((rule) => rule.source === "/api/:path*");
      expect(apiRule).toBeDefined();
      expect(apiRule!.destination).toMatch(/http:\/\/localhost:8000/);

      if (saved !== undefined) {
        process.env.API_URL = saved;
      }
      if (savedPublic !== undefined) {
        process.env.NEXT_PUBLIC_API_URL = savedPublic;
      }
    });
  });

  it("uses NEXT_PUBLIC_API_URL when set", () => {
    const saved = process.env.API_URL;
    const savedPublic = process.env.NEXT_PUBLIC_API_URL;
    process.env.API_URL = "https://api.example.com";
    process.env.NEXT_PUBLIC_API_URL = "https://public-api.example.com";

    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const freshConfig = require("../next.config.js");
    const rules: RewriteRule[] | Promise<RewriteRule[]> = freshConfig.rewrites();
    const resolved = rules instanceof Promise ? rules : Promise.resolve(rules);

    return resolved.then((r) => {
      const apiRule = r.find((rule) => rule.source === "/api/:path*");
      expect(apiRule).toBeDefined();
      expect(apiRule!.destination).toMatch(/https:\/\/public-api\.example\.com/);

      if (saved !== undefined) {
        process.env.API_URL = saved;
      } else {
        delete process.env.API_URL;
      }
      if (savedPublic !== undefined) {
        process.env.NEXT_PUBLIC_API_URL = savedPublic;
      } else {
        delete process.env.NEXT_PUBLIC_API_URL;
      }
    });
  });

  it("uses API_URL when set", () => {
    const saved = process.env.API_URL;
    const savedPublic = process.env.NEXT_PUBLIC_API_URL;
    process.env.API_URL = "https://api.example.com";
    delete process.env.NEXT_PUBLIC_API_URL;

    jest.resetModules();
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const freshConfig = require("../next.config.js");
    const rules: RewriteRule[] | Promise<RewriteRule[]> = freshConfig.rewrites();
    const resolved = rules instanceof Promise ? rules : Promise.resolve(rules);

    return resolved.then((r) => {
      const apiRule = r.find((rule) => rule.source === "/api/:path*");
      expect(apiRule).toBeDefined();
      expect(apiRule!.destination).toMatch(/https:\/\/api\.example\.com/);

      if (saved !== undefined) {
        process.env.API_URL = saved;
      } else {
        delete process.env.API_URL;
      }
      if (savedPublic !== undefined) {
        process.env.NEXT_PUBLIC_API_URL = savedPublic;
      }
    });
  });

  it("rewrites preserve the path wildcard segment", () => {
    const apiRule = rewrites.find((r) => r.source === "/api/:path*");
    expect(apiRule).toBeDefined();
    expect(apiRule!.source).toContain(":path*");
    expect(apiRule!.destination).toContain(":path*");
  });
});
