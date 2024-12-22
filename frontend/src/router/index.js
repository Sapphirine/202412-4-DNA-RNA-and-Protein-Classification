import Vue from 'vue'
import Router from 'vue-router'
import Layout from '@/layout/index'
import NProgress from 'nprogress' // progress bar
import 'nprogress/nprogress.css' // progress bar style
import Config from '@/settings'
import { getToken } from '@/utils/auth' // getToken from cookie

Vue.use(Router)

// Define static routes for the application
export const staticRoutes = [
  {
    path: '/',
    component: Layout,
    redirect: '/dashboard', // Redirect to dashboard by default
    children: [
      {
        path: 'dashboard',
        component: (resolve) => require(['@/views/home'], resolve),
        name: 'Dashboard',
        meta: { title: 'Dashboard', icon: 'index', affix: true, noCache: true }
      }
    ]
  },
  {
    path: '/features',
    component: Layout,
    meta: { title: 'Function', icon: 'el-icon-folder' },
    children: [
      {
        path: 'drug-category',
        component: () => import('@/views/DrugCategory.vue'), // Lazy-load the classification component
        meta: { title: 'Classification', icon: 'el-icon-document' }
      }
    ]
  }
]

// Create a new router instance
const router = new Router({
  mode: 'history', // Use history mode for cleaner URLs
  scrollBehavior: () => ({ y: 0 }), // Scroll to the top on navigation
  routes: staticRoutes
})

// Navigation guards
router.beforeEach((to, from, next) => {
  // Update document title based on the route meta title
  if (to.meta.title) {
    document.title = `${to.meta.title} - ${Config.title}`
  }
  NProgress.start() // Start progress bar

  // Check if user is authenticated
  if (getToken()) {
    next()
  } else {
    // Redirect to login if not authenticated
    if (whiteList.includes(to.path)) {
      next()
    } else {
      next(`/login?redirect=${to.fullPath}`)
      NProgress.done() // End progress bar
    }
  }
})

router.afterEach(() => {
  NProgress.done() // End progress bar after navigation
})

export default router
