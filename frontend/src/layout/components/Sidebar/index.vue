<template>
  <div :class="{ 'has-logo': showLogo }">
    <!-- Sidebar logo -->
    <logo v-if="showLogo" :collapse="isCollapse" />
    <!-- Scrollable menu container -->
    <el-scrollbar wrap-class="scrollbar-wrapper">
      <!-- Sidebar menu -->
      <el-menu
        :default-active="activeMenu"
        :collapse="isCollapse"
        :background-color="variables.menuBg"
        :text-color="variables.menuText"
        :unique-opened="$store.state.settings.uniqueOpened"
        :active-text-color="variables.menuActiveText"
        :collapse-transition="false"
        mode="vertical"
      >
        <!-- Iterate over menu items to render sidebar links -->
        <sidebar-item
          v-for="route in menuItems"
          :key="route.path"
          :item="route"
          :base-path="route.path"
        />
      </el-menu>
    </el-scrollbar>
  </div>
</template>

<script>
import { menuItems } from '@/menuConfig' // Import menu configuration
import { mapGetters } from 'vuex'
import Logo from './Logo'
import SidebarItem from './SidebarItem'
import variables from '@/assets/styles/variables.scss' // Import theme variables

export default {
  components: { SidebarItem, Logo },
  data() {
    return {
      menuItems // Static menu items for the sidebar
    }
  },
  computed: {
    ...mapGetters(['sidebar']), // Access Vuex state for sidebar settings
    activeMenu() {
      // Determine the currently active menu item
      const { meta, path } = this.$route
      return meta.activeMenu || path
    },
    showLogo() {
      // Control logo visibility based on store settings
      return this.$store.state.settings.sidebarLogo
    },
    variables() {
      return variables // Bind theme variables for styling
    },
    isCollapse() {
      // Determine sidebar collapsed state
      return !this.sidebar.opened
    }
  }
}
</script>
