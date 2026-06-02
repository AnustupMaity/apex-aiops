"use client";

import React from "react";

interface QueryDiffProps {
  original: string;
  optimized: string | null;
}

export default function QueryDiff({ original, optimized }: QueryDiffProps) {
  return (
    <div className="sql-diff" id="query-diff-viewer">
      <div className="sql-diff-panel">
        <div className="sql-diff-header original">⬤ Original Query</div>
        <div className="sql-diff-body">{original}</div>
      </div>
      <div className="sql-diff-panel">
        <div className="sql-diff-header optimized">⬤ Optimized Query</div>
        <div className="sql-diff-body">
          {optimized || "No optimization available"}
        </div>
      </div>
    </div>
  );
}
