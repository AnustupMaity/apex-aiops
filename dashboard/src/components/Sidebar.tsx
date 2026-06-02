"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, AlertCircle, BarChart3, Settings, Terminal } from "lucide-react";

const navItems = [
  {
    section: "Monitoring",
    items: [
      { label: "Dashboard", href: "/", icon: <LayoutDashboard size={18} /> },
      { label: "Incidents", href: "/incidents", icon: <AlertCircle size={18} /> },
    ],
  },
  {
    section: "System",
    items: [
      { label: "Metrics", href: "/metrics", icon: <BarChart3 size={18} /> },
      { label: "Settings", href: "/settings", icon: <Settings size={18} /> },
    ],
  },
  {
    section: "Tools",
    items: [
      { label: "Diagnostic Console", href: "/test", icon: <Terminal size={18} /> },
    ],
  },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar" id="sidebar-nav">
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="logo-icon">A</div>
        <div>
          <h1>Apex Core</h1>
          <span>Autonomous Database Engine</span>
        </div>
      </div>

      {/* Navigation */}
      <nav>
        {navItems.map((section) => (
          <div className="nav-section" key={section.section}>
            <div className="nav-section-title">{section.section}</div>
            {section.items.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className={`nav-link ${
                  pathname === item.href ? "active" : ""
                }`}
                id={`nav-${item.label.toLowerCase().replace(' ', '-')}`}
              >
                <span className="nav-icon" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {item.icon}
                </span>
                {item.label}
              </Link>
            ))}
          </div>
        ))}
      </nav>

      {/* Status Footer */}
      <div style={{ marginTop: "auto", padding: "16px 12px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "8px",
            fontSize: "12px",
            color: "var(--text-secondary)",
          }}
        >
          <span className="status-dot online"></span>
          System Online
        </div>
      </div>
    </aside>
  );
}
