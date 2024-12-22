import Vue from 'vue'
import Router from 'vue-router'
import Layout from '../layout/index'

Vue.use(Router)

// Define constant routes that are always available
export const constantRouterMap = [
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard', // Redirect to dashboard as the default route
    children: [
      {
        path: 'dashboard',
        component: (resolve) => require(['@/views/home'], resolve),
        name: 'Dashboard',
        meta: { title: 'Dashboard', icon: 'index', affix: true, noCache: true } // Metadata for the dashboard
      }
    ]
  },
  {
    path: '/user',
    component: Layout,
    hidden: true, // Hide this route from sidebar navigation
    redirect: 'noredirect', // Prevent unnecessary redirection
    children: [
      {
        path: '/drug-category',
        name: 'DrugCategory',
        component: () => import('@/views/DrugCategory.vue'), // Lazy-load the classification component
        meta: { title: 'Classification', showInSidebar: true } // Metadata for classification
      }
    ]
  }
]

// Create and export the router instance
export default new Router({
  // Use history mode for cleaner URLs (commented 'hash' mode available for fallback)
  mode: 'history',
  scrollBehavior: () => ({ y: 0 }), // Ensure scrolling resets to top for every navigation
  routes: constantRouterMap
})
