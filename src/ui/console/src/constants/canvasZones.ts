/**
 * Shared constants for canvas zones and element classification.
 * Used by ReviewCanvas, ElementList, VirtualModelRenderer, etc.
 */

/** Region IDs that represent dynamic content areas (chat messages, documents, etc.) */
export const DYNAMIC_CONTENT_ZONES = new Set([
  'message_area', 'content_stream', 'document_body',
  'chat_message_area', 'right_panel_middle',
  'article_body', 'code_block', 'list_dynamic_content',
]);

/** Semantic roles that represent fixed UI controls (not dynamic content) */
export const FIXED_CONTROL_ROLES = new Set([
  'button', 'send_button', 'submit_button', 'cancel_button',
  'icon_button', 'toggle_button', 'message_input', 'text_input',
  'search_input', 'password_input', 'file_input', 'tab',
  'nav_item', 'sidebar', 'toolbar', 'menu_bar', 'title_bar', 'status_bar',
  'menu_item',
]);
