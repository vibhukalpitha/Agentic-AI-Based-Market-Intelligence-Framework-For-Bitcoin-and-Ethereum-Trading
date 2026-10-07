import React from 'react';
import type { Component2SubTab } from '../types';

interface SidebarProps {
  activePage: Component2SubTab;
  setActivePage: (page: Component2SubTab) => void;
}

export const Sidebar: React.FC<SidebarProps> = () => {
  return null;
};
