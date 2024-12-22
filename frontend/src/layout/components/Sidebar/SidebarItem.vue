<template>
  <div class="menu-wrapper">
    <!-- Submenu container for "Function" -->
    <el-submenu index="1" popper-append-to-body>
      <template slot="title">
        <!-- Render the submenu title with an icon and text -->
        <item icon="el-icon-folder" title="Function" />
      </template>
      <!-- Menu item for "Classification" -->
      <el-menu-item index="1-1">
        <!-- Use AppLink to navigate to "drug-category" -->
        <app-link to="/drug-category">
          <item icon="el-icon-document" title="Classification" />
        </app-link>
      </el-menu-item>
    </el-submenu>
  </div>
</template>

<script>
import Item from './Item' // Menu item component
import AppLink from './Link' // Link wrapper for navigation

export default {
  name: 'SidebarItem',
  components: { Item, AppLink } // Register child components
}
</script>
